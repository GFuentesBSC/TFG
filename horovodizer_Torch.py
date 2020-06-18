from ast import *
import os
import astunparse as au
import inspect
from hvdconfig import Torch as hvdconfig
import time
import subprocess
import ntpath
from subprocess import check_output
from copy import deepcopy as copy

model_names = set()

class RootNodeException(Exception):
    """ raised when a node is not a root node """
    pass

def classname(cls):
    return cls.__class__.__name__.lower()

def get_body_nodes(root_node):
    ret_list = []
    try:
        body_node = getattr(root_node, 'body')
        ret_list.append(body_node)
        orelse_node = getattr(root_node, 'orelse')
        ret_list.append(orelse_node)
        finally_node = getattr(root_node, 'finalbody')
        ret_list.append(finally_node)
    except Exception:
        pass
    return ret_list

def add_code(body, index, extra_code):
    for elem in extra_code:
        body.insert(index, elem)
        index += 1
    return index

def parse_code(filename, save_to_file):
    original_code = open(filename, 'r').read()
    root_node = parse(original_code)
    if save_to_file:
        filename2 = "parsed/parsed_" + ntpath.basename(filename)
        with open(filename2, 'w') as file:
            file.write(au.dump(root_node))
    return root_node

def recover_code_from_ast_file(filename, save_to_file):
    parsed_code = open(filename, 'r').read()
    global node
    code_to_exec = "node = " + parsed_code
    exec(code_to_exec, globals())
    return generate_code_from_ast(node, save_to_file)

def generate_horovodized_code(root_node, filename):

    directory = os.path.join('PARALLEL', 'TORCH')
    filename = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    python_code = au.unparse(root_node)

    file = open(filename, 'w')
    file.write(python_code)
    file.close()
    try:
        import black
        out = check_output(["black", filename], stderr=subprocess.STDOUT)
    except ModuleNotFoundError:
        print("Could not format code! Run pip install black")
    return filename

def id_or_attr(node, name):

    correct = False
    try:
        correct = node.value.func.attr.lower() == name.lower()
    except AttributeError:
        try:
            correct = node.value.func.id.lower() == name.lower()
        except AttributeError:
            pass

    return correct

def add_import(root_node, code):
    name = code.names[0].name
    last_idx = 0
    exists = False
    hvd = [code]
    body = get_body_nodes(root_node)[0]
    for idx, elem in enumerate(body):
        if classname(elem).startswith('import'):
            last_idx = idx
            if elem.names[0].name == name:
                if VERBOSE:
                    print(name + ' import already exists')
                exists = True
        else:
            if not exists:
                last_idx = add_code(body, idx, hvd)
            break

    return last_idx

def add_horovod_initialization(root_node, idx):

    body = get_body_nodes(root_node)[0]
    return add_code(body, idx, hvdconfig.configs)

def find_names_in_code(code):
    return [x for id, x in enumerate(code.split('\'')) if id%2 == 1]

def find_variables_in_code(var_names, body, idx1):
    elems = list()
    idxs = list()
    for idx, el in enumerate(body):
        if idx > idx1:
            names = find_names_in_code(au.dump(el))
            for n in var_names:
                if n in names:
                    elems.append(el)
                    names = []
                    idxs.append(idx)
    for i in reversed(idxs):
        del body[i]
    return elems

def find_model_name(node):

    model_name = 'model'
    params_kw = False
    try:
        for kw in node.value.keywords:
            if kw.arg == 'params':
                params_kw = True
                model_name = kw.value.func.value.id
        if not params_kw:
            # expects first parameter to be like: model.parameters()
            model_name = node.value.args[0].func.value.id
    except AttributeError:
        pass

    return model_name

def adapt_optimizer(root_node):

    opt_types = hvdconfig.possible_optim_names_torchvision
    model_name = 'model'
    opt_name = 'optimizer'

    def adapt_optimizer_recursive(body, adapted):
        found = False
        for idx, elem in enumerate(body):
            found = False
            if classname(elem) == 'assign':
                try:
                    if elem.value.func.id in opt_types:
                        found = True
                except AttributeError:
                    try:
                        if elem.value.func.attr in opt_types:
                            found = True
                    except AttributeError:
                        pass
                if found:
                    opt_name = elem.targets[0].id
                    model_name = find_model_name(elem)
                    model_names.add(model_name)
                    node1 = copy(hvdconfig.adapt_opt)
                    node1.value.args[0].id = opt_name
                    node1.value.args[1].id = model_name
                    add_code(body, idx+1, [node1])
                    adapted = True
            else:
                for body2 in get_body_nodes(elem):
                    adapted, found2 = adapt_optimizer_recursive(body2, adapted)
                    found = found or found2
        return adapted, found

    body = get_body_nodes(root_node)[0]
    adapted, found = adapt_optimizer_recursive(body, False)

    if not adapted and VERBOSE:
        print("Could not adapt optimizer!")
        if not found:
            print("-> Optimizer initialization not found")

def adapt_model(root_node):

    def model_cuda(body, model_name):
        for idx, elem in enumerate(body):
            try:
                if classname(elem) == 'expr':
                    if elem.value.func.value.id == model_name and elem.value.func.attr == 'cuda':
                        return True
            except AttributeError:
                pass

        return False

    def adapt_model_recursive(body, adapted, model_name, new_node):
        found = False
        for idx, elem in enumerate(body):
            try:
                if classname(elem) == 'assign':
                    # add model.cuda()
                    if elem.targets[0].id == model_name:
                        found = True
                        # check if model.cuda() is already in code
                        if not model_cuda(body, model_name):
                            add_code(body, idx + 1, [new_node])
                        adapted = True
                elif classname(elem) == 'expr':
                    # remove model.to(device)
                    if elem.value.func.value.id in model_names and elem.value.func.attr == 'to':
                        del body[idx]
            except AttributeError:
                pass
            else:
                for body2 in get_body_nodes(elem):
                    adapted, found2 = adapt_model_recursive(body2, adapted, model_name, new_node)
                    found = found or found2
        return adapted, found

    def broadcast_parameters_recursive(body, new_node, model_name):
        for idx, elem in enumerate(body):
            try:
                if classname(elem) == 'expr':
                    # add model.cuda()
                    if elem.value.func.value.id == model_name and elem.value.func.attr == 'cuda':
                        add_code(body, idx + 1, [new_node])
                        break
            except AttributeError:
                pass
            else:
                for body2 in get_body_nodes(elem):
                    adapt_model_recursive(body2, adapted, model_name, new_node)

    body = get_body_nodes(root_node)[0]
    for model_name in model_names:
        new_node = copy(hvdconfig.model_to_cuda)
        new_node.value.func.value.id = model_name
        adapted, found = adapt_model_recursive(body, False, model_name, new_node)
        if not adapted:
            if VERBOSE:
                print("Could not adapt model: " + model_name + " to cuda()!")
                if not found:
                    print("-> model: '" + model_name + "' initialization not found")
        else:
            node = copy(hvdconfig.broadcast_parameters)
            node.value.args[0].func.value.id = model_name
            broadcast_parameters_recursive(body, node, model_name)

def adapt_model_save(root_node):

    def adapt_model_save_recursive(body, adapted, model_names):
        found = False
        for idx1, elem in enumerate(body):
            try:
                if classname(elem) == 'expr' and \
                    elem.value.func.attr in ['save', 'save_weights'] and \
                        (elem.value.func.value.id in model_names or 'model' in elem.value.func.value.id):
                    found = True
                    node = copy(hvdconfig.if_rank_0)
                    node.body = [elem]
                    body[idx1] = node
                    adapted = True
                else:
                    for body2 in get_body_nodes(elem):
                        adapted, found2 = adapt_model_save_recursive(body2, adapted, model_names)
                        found = found or found2
            except AttributeError:
                pass
        return adapted, found

    body = get_body_nodes(root_node)[0]
    adapted, found = adapt_model_save_recursive(body, False, model_names)
    if not adapted and VERBOSE:
        print("Could not adapt model.save() function!")
        if not found:
            print("    --Function not found--")

def adapt_data_loaders(root_node):

    def adapt_data_loader_recursive(body, adapted):
        found = False
        for idx, elem in enumerate(body):
            if not found:
                try:
                    if classname(elem) == 'assign' and id_or_attr(elem, 'DataLoader'):
                        keywords = elem.value.keywords
                        if 'sampler' not in [kw.arg for kw in keywords]:
                            found = True
                            dataset_name = 'dataset'
                            for kw in keywords:
                                if kw.arg == 'dataset':
                                    dataset_name = kw.value.id
                                    break
                            else:
                                if len(elem.value.args) > 0:
                                    dataset_name = elem.value.args[0].id
                                else:
                                    break
                            new_node = copy(hvdconfig.data_sampler)
                            new_kw = copy(hvdconfig.data_sampler_keyword)
                            sampler_name = str(new_node.targets[0].id + elem.targets[0].id)
                            new_node.targets[0].id = new_kw.value.id = sampler_name
                            new_node.value.keywords[0].value.id = dataset_name
                            keywords.append(new_kw)
                            add_code(body, idx, [new_node])
                            adapted = True

                    else:
                        for body2 in get_body_nodes(elem):
                            adapted, found2 = adapt_data_loader_recursive(body2, adapted)
                            found = found or found2
                except AttributeError:
                    pass
            else:
                found = False
        return adapted, found

    body = get_body_nodes(root_node)[0]
    adapted, found = adapt_data_loader_recursive(body, False)
    if not adapted and VERBOSE:
        print("Could not adapt train_data_loader!")
        if not found:
            print(" Data loader not found--")

def remove_block_comments(root_node):

    def remove_block_comments_recursive(body):
        sub_index_list = []
        for idx1, elem in enumerate(body):
            try:
                if classname(elem) == 'expr' and classname(elem.value) == 'str':
                    sub_index_list.append(idx1)
                elif not classname(elem) == 'assign':
                    for body2 in get_body_nodes(elem):
                        remove_block_comments_recursive(body2)
            except AttributeError:
                pass
        for i in reversed(sub_index_list):
            del body[i]

    remove_block_comments_recursive(get_body_nodes(root_node)[0])

def add_auxiliar_functions(root_node, index):
    body = getattr(root_node, 'body')
    for aux_func in hvdconfig.aux_funcs:
        body.insert(index, aux_func)

def add_necessary_imports(node):
    last_idx = 0
    for imp in hvdconfig.imports:
        idx = add_import(node, imp)
        if idx > last_idx:
            last_idx = idx
    return last_idx

def horovodize(path, v):

    global VERBOSE
    VERBOSE = v

    # filename of horovod code
    name = "hvd_" + ntpath.basename(path)

    # Obtain AST from original code
    node = parse_code(filename=path, save_to_file=True)

    # Remove block comments
    remove_block_comments(node)
    print("✓ Removed unnecesary comments")

    # Add necessary imports
    last_idx = add_necessary_imports(node)
    print("✓ Added imports")

    # Add horovod config
    add_horovod_initialization(node, last_idx)
    print("✓ Added Horovod init")

    # Add auxiliar functions used by horovod
    add_auxiliar_functions(node, last_idx)
    print("✓ Added auxiliar functions")

    # Adapt data loaders (using data DistributedSamplers)
    adapt_data_loaders(node)

    # Adapt optimizer (hvd.distributedOptimizer)
    adapt_optimizer(node)

    # Send model to cuda
    # Broadcast parameters from rank 0 to all other processes
    adapt_model(node)
    print("✓ Code completely adapted")

    # Generate new python code
    filename = generate_horovodized_code(root_node=node, filename=name)
    print(f"✓ File generated: {filename}")
