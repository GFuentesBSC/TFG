Module(body=[
  Import(names=[alias(
    name='torch',
    asname=None)]),
  Import(names=[alias(
    name='torchvision',
    asname=None)]),
  Import(names=[alias(
    name='torchvision.transforms',
    asname='transforms')]),
  Import(names=[alias(
    name='torch.nn',
    asname='nn')]),
  Import(names=[alias(
    name='torch.nn.functional',
    asname='F')]),
  Import(names=[alias(
    name='torch.optim',
    asname='optim')]),
  Assign(
    targets=[Name(
      id='classes',
      ctx=Store())],
    value=Tuple(
      elts=[
        Str(s='plane'),
        Str(s='car'),
        Str(s='bird'),
        Str(s='cat'),
        Str(s='deer'),
        Str(s='dog'),
        Str(s='frog'),
        Str(s='horse'),
        Str(s='ship'),
        Str(s='truck')],
      ctx=Load())),
  Assign(
    targets=[Name(
      id='batch_size',
      ctx=Store())],
    value=Num(n=32)),
  Assign(
    targets=[Name(
      id='num_classes',
      ctx=Store())],
    value=Num(n=10)),
  Assign(
    targets=[Name(
      id='epochs',
      ctx=Store())],
    value=Num(n=100)),
  Assign(
    targets=[Name(
      id='learning_rate',
      ctx=Store())],
    value=Num(n=0.001)),
  Assign(
    targets=[Name(
      id='momentum',
      ctx=Store())],
    value=Num(n=0.9)),
  Assign(
    targets=[Name(
      id='transform',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='transforms',
          ctx=Load()),
        attr='Compose',
        ctx=Load()),
      args=[List(
        elts=[
          Call(
            func=Attribute(
              value=Name(
                id='transforms',
                ctx=Load()),
              attr='ToTensor',
              ctx=Load()),
            args=[],
            keywords=[]),
          Call(
            func=Attribute(
              value=Name(
                id='transforms',
                ctx=Load()),
              attr='Normalize',
              ctx=Load()),
            args=[
              Tuple(
                elts=[
                  Num(n=0.5),
                  Num(n=0.5),
                  Num(n=0.5)],
                ctx=Load()),
              Tuple(
                elts=[
                  Num(n=0.5),
                  Num(n=0.5),
                  Num(n=0.5)],
                ctx=Load())],
            keywords=[])],
        ctx=Load())],
      keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[Str(s='Downloading dataset...')],
    keywords=[])),
  Assign(
    targets=[Name(
      id='trainset',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Name(
            id='torchvision',
            ctx=Load()),
          attr='datasets',
          ctx=Load()),
        attr='CIFAR10',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='root',
          value=Str(s='~/.keras/datasets')),
        keyword(
          arg='train',
          value=NameConstant(value=True)),
        keyword(
          arg='download',
          value=NameConstant(value=True)),
        keyword(
          arg='transform',
          value=Name(
            id='transform',
            ctx=Load()))])),
  Assign(
    targets=[Name(
      id='trainloader',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Attribute(
            value=Name(
              id='torch',
              ctx=Load()),
            attr='utils',
            ctx=Load()),
          attr='data',
          ctx=Load()),
        attr='DataLoader',
        ctx=Load()),
      args=[Name(
        id='trainset',
        ctx=Load())],
      keywords=[
        keyword(
          arg='batch_size',
          value=Name(
            id='batch_size',
            ctx=Load())),
        keyword(
          arg='shuffle',
          value=NameConstant(value=True)),
        keyword(
          arg='num_workers',
          value=Num(n=2))])),
  Assign(
    targets=[Name(
      id='testset',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Name(
            id='torchvision',
            ctx=Load()),
          attr='datasets',
          ctx=Load()),
        attr='CIFAR10',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='root',
          value=Str(s='~/.keras/datasets')),
        keyword(
          arg='train',
          value=NameConstant(value=False)),
        keyword(
          arg='download',
          value=NameConstant(value=True)),
        keyword(
          arg='transform',
          value=Name(
            id='transform',
            ctx=Load()))])),
  Assign(
    targets=[Name(
      id='testloader',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Attribute(
            value=Name(
              id='torch',
              ctx=Load()),
            attr='utils',
            ctx=Load()),
          attr='data',
          ctx=Load()),
        attr='DataLoader',
        ctx=Load()),
      args=[Name(
        id='testset',
        ctx=Load())],
      keywords=[
        keyword(
          arg='batch_size',
          value=Name(
            id='batch_size',
            ctx=Load())),
        keyword(
          arg='shuffle',
          value=NameConstant(value=False)),
        keyword(
          arg='num_workers',
          value=Num(n=2))])),
  Assign(
    targets=[Name(
      id='net',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='nn',
          ctx=Load()),
        attr='Sequential',
        ctx=Load()),
      args=[
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Conv2d',
            ctx=Load()),
          args=[
            Num(n=3),
            Num(n=64)],
          keywords=[
            keyword(
              arg='kernel_size',
              value=Num(n=3)),
            keyword(
              arg='padding',
              value=Num(n=1))]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Conv2d',
            ctx=Load()),
          args=[
            Num(n=64),
            Num(n=64)],
          keywords=[
            keyword(
              arg='kernel_size',
              value=Num(n=3)),
            keyword(
              arg='padding',
              value=Num(n=1))]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='MaxPool2d',
            ctx=Load()),
          args=[Num(n=2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Conv2d',
            ctx=Load()),
          args=[
            Num(n=64),
            Num(n=1024)],
          keywords=[
            keyword(
              arg='kernel_size',
              value=Num(n=3)),
            keyword(
              arg='padding',
              value=Num(n=1))]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Conv2d',
            ctx=Load()),
          args=[
            Num(n=1024),
            Num(n=1024)],
          keywords=[
            keyword(
              arg='kernel_size',
              value=Num(n=3)),
            keyword(
              arg='padding',
              value=Num(n=1))]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='MaxPool2d',
            ctx=Load()),
          args=[Num(n=2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Conv2d',
            ctx=Load()),
          args=[
            Num(n=1024),
            Num(n=1024)],
          keywords=[
            keyword(
              arg='kernel_size',
              value=Num(n=3)),
            keyword(
              arg='padding',
              value=Num(n=1))]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Conv2d',
            ctx=Load()),
          args=[
            Num(n=1024),
            Num(n=1024)],
          keywords=[
            keyword(
              arg='kernel_size',
              value=Num(n=3)),
            keyword(
              arg='padding',
              value=Num(n=1))]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='MaxPool2d',
            ctx=Load()),
          args=[Num(n=2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Flatten',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Linear',
            ctx=Load()),
          args=[
            Num(n=16384),
            Num(n=128)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='ReLU',
            ctx=Load()),
          args=[],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[]),
        Call(
          func=Attribute(
            value=Name(
              id='nn',
              ctx=Load()),
            attr='Linear',
            ctx=Load()),
          args=[
            Num(n=128),
            Num(n=10)],
          keywords=[])],
      keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='net',
        ctx=Load()),
      attr='cuda',
      ctx=Load()),
    args=[],
    keywords=[])),
  Assign(
    targets=[Name(
      id='criterion',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='nn',
          ctx=Load()),
        attr='CrossEntropyLoss',
        ctx=Load()),
      args=[],
      keywords=[])),
  Assign(
    targets=[Name(
      id='optimizer',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='optim',
          ctx=Load()),
        attr='SGD',
        ctx=Load()),
      args=[Call(
        func=Attribute(
          value=Name(
            id='net',
            ctx=Load()),
          attr='parameters',
          ctx=Load()),
        args=[],
        keywords=[])],
      keywords=[
        keyword(
          arg='lr',
          value=Name(
            id='learning_rate',
            ctx=Load())),
        keyword(
          arg='momentum',
          value=Name(
            id='momentum',
            ctx=Load()))])),
  FunctionDef(
    name='train',
    args=arguments(
      args=[arg(
        arg='epoch',
        annotation=None)],
      vararg=None,
      kwonlyargs=[],
      kw_defaults=[],
      kwarg=None,
      defaults=[]),
    body=[
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='net',
            ctx=Load()),
          attr='train',
          ctx=Load()),
        args=[],
        keywords=[])),
      Assign(
        targets=[Name(
          id='train_loss',
          ctx=Store())],
        value=Num(n=0)),
      Assign(
        targets=[Name(
          id='correct',
          ctx=Store())],
        value=Num(n=0)),
      Assign(
        targets=[Name(
          id='total',
          ctx=Store())],
        value=Num(n=0)),
      For(
        target=Tuple(
          elts=[
            Name(
              id='batch_idx',
              ctx=Store()),
            Tuple(
              elts=[
                Name(
                  id='inputs',
                  ctx=Store()),
                Name(
                  id='targets',
                  ctx=Store())],
              ctx=Store())],
          ctx=Store()),
        iter=Call(
          func=Name(
            id='enumerate',
            ctx=Load()),
          args=[Name(
            id='trainloader',
            ctx=Load())],
          keywords=[]),
        body=[
          Assign(
            targets=[Tuple(
              elts=[
                Name(
                  id='inputs',
                  ctx=Store()),
                Name(
                  id='targets',
                  ctx=Store())],
              ctx=Store())],
            value=Tuple(
              elts=[
                Call(
                  func=Attribute(
                    value=Name(
                      id='inputs',
                      ctx=Load()),
                    attr='cuda',
                    ctx=Load()),
                  args=[],
                  keywords=[]),
                Call(
                  func=Attribute(
                    value=Name(
                      id='targets',
                      ctx=Load()),
                    attr='cuda',
                    ctx=Load()),
                  args=[],
                  keywords=[])],
              ctx=Load())),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='optimizer',
                ctx=Load()),
              attr='zero_grad',
              ctx=Load()),
            args=[],
            keywords=[])),
          Assign(
            targets=[Name(
              id='outputs',
              ctx=Store())],
            value=Call(
              func=Name(
                id='net',
                ctx=Load()),
              args=[Name(
                id='inputs',
                ctx=Load())],
              keywords=[])),
          Assign(
            targets=[Name(
              id='loss',
              ctx=Store())],
            value=Call(
              func=Name(
                id='criterion',
                ctx=Load()),
              args=[
                Name(
                  id='outputs',
                  ctx=Load()),
                Name(
                  id='targets',
                  ctx=Load())],
              keywords=[])),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='loss',
                ctx=Load()),
              attr='backward',
              ctx=Load()),
            args=[],
            keywords=[])),
          Expr(value=Call(
            func=Attribute(
              value=Name(
                id='optimizer',
                ctx=Load()),
              attr='step',
              ctx=Load()),
            args=[],
            keywords=[])),
          AugAssign(
            target=Name(
              id='train_loss',
              ctx=Store()),
            op=Add(),
            value=Call(
              func=Attribute(
                value=Name(
                  id='loss',
                  ctx=Load()),
                attr='item',
                ctx=Load()),
              args=[],
              keywords=[])),
          Assign(
            targets=[Tuple(
              elts=[
                Name(
                  id='_',
                  ctx=Store()),
                Name(
                  id='predicted',
                  ctx=Store())],
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Name(
                  id='outputs',
                  ctx=Load()),
                attr='max',
                ctx=Load()),
              args=[Num(n=1)],
              keywords=[])),
          AugAssign(
            target=Name(
              id='total',
              ctx=Store()),
            op=Add(),
            value=Call(
              func=Attribute(
                value=Name(
                  id='targets',
                  ctx=Load()),
                attr='size',
                ctx=Load()),
              args=[Num(n=0)],
              keywords=[])),
          AugAssign(
            target=Name(
              id='correct',
              ctx=Store()),
            op=Add(),
            value=Call(
              func=Attribute(
                value=Call(
                  func=Attribute(
                    value=Call(
                      func=Attribute(
                        value=Name(
                          id='predicted',
                          ctx=Load()),
                        attr='eq',
                        ctx=Load()),
                      args=[Name(
                        id='targets',
                        ctx=Load())],
                      keywords=[]),
                    attr='sum',
                    ctx=Load()),
                  args=[],
                  keywords=[]),
                attr='item',
                ctx=Load()),
              args=[],
              keywords=[]))],
        orelse=[]),
      Assign(
        targets=[Name(
          id='final_loss',
          ctx=Store())],
        value=Call(
          func=Name(
            id='round',
            ctx=Load()),
          args=[
            BinOp(
              left=Name(
                id='train_loss',
                ctx=Load()),
              op=Div(),
              right=BinOp(
                left=Name(
                  id='batch_idx',
                  ctx=Load()),
                op=Add(),
                right=Num(n=1))),
            Num(n=2)],
          keywords=[])),
      Assign(
        targets=[Name(
          id='final_acc',
          ctx=Store())],
        value=Call(
          func=Name(
            id='round',
            ctx=Load()),
          args=[
            BinOp(
              left=BinOp(
                left=Num(n=100),
                op=Mult(),
                right=Name(
                  id='correct',
                  ctx=Load())),
              op=Div(),
              right=Name(
                id='total',
                ctx=Load())),
            Num(n=2)],
          keywords=[])),
      Return(value=Tuple(
        elts=[
          Name(
            id='final_loss',
            ctx=Load()),
          Name(
            id='final_acc',
            ctx=Load())],
        ctx=Load()))],
    decorator_list=[],
    returns=None),
  FunctionDef(
    name='test',
    args=arguments(
      args=[arg(
        arg='epoch',
        annotation=None)],
      vararg=None,
      kwonlyargs=[],
      kw_defaults=[],
      kwarg=None,
      defaults=[]),
    body=[
      Global(names=['best_acc']),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='net',
            ctx=Load()),
          attr='eval',
          ctx=Load()),
        args=[],
        keywords=[])),
      Assign(
        targets=[Name(
          id='test_loss',
          ctx=Store())],
        value=Num(n=0)),
      Assign(
        targets=[Name(
          id='correct',
          ctx=Store())],
        value=Num(n=0)),
      Assign(
        targets=[Name(
          id='total',
          ctx=Store())],
        value=Num(n=0)),
      With(
        items=[withitem(
          context_expr=Call(
            func=Attribute(
              value=Name(
                id='torch',
                ctx=Load()),
              attr='no_grad',
              ctx=Load()),
            args=[],
            keywords=[]),
          optional_vars=None)],
        body=[
          For(
            target=Tuple(
              elts=[
                Name(
                  id='batch_idx',
                  ctx=Store()),
                Tuple(
                  elts=[
                    Name(
                      id='inputs',
                      ctx=Store()),
                    Name(
                      id='targets',
                      ctx=Store())],
                  ctx=Store())],
              ctx=Store()),
            iter=Call(
              func=Name(
                id='enumerate',
                ctx=Load()),
              args=[Name(
                id='testloader',
                ctx=Load())],
              keywords=[]),
            body=[
              Assign(
                targets=[Tuple(
                  elts=[
                    Name(
                      id='inputs',
                      ctx=Store()),
                    Name(
                      id='targets',
                      ctx=Store())],
                  ctx=Store())],
                value=Tuple(
                  elts=[
                    Call(
                      func=Attribute(
                        value=Name(
                          id='inputs',
                          ctx=Load()),
                        attr='cuda',
                        ctx=Load()),
                      args=[],
                      keywords=[]),
                    Call(
                      func=Attribute(
                        value=Name(
                          id='targets',
                          ctx=Load()),
                        attr='cuda',
                        ctx=Load()),
                      args=[],
                      keywords=[])],
                  ctx=Load())),
              Assign(
                targets=[Name(
                  id='outputs',
                  ctx=Store())],
                value=Call(
                  func=Name(
                    id='net',
                    ctx=Load()),
                  args=[Name(
                    id='inputs',
                    ctx=Load())],
                  keywords=[])),
              Assign(
                targets=[Name(
                  id='loss',
                  ctx=Store())],
                value=Call(
                  func=Name(
                    id='criterion',
                    ctx=Load()),
                  args=[
                    Name(
                      id='outputs',
                      ctx=Load()),
                    Name(
                      id='targets',
                      ctx=Load())],
                  keywords=[])),
              AugAssign(
                target=Name(
                  id='test_loss',
                  ctx=Store()),
                op=Add(),
                value=Call(
                  func=Attribute(
                    value=Name(
                      id='loss',
                      ctx=Load()),
                    attr='item',
                    ctx=Load()),
                  args=[],
                  keywords=[])),
              Assign(
                targets=[Tuple(
                  elts=[
                    Name(
                      id='_',
                      ctx=Store()),
                    Name(
                      id='predicted',
                      ctx=Store())],
                  ctx=Store())],
                value=Call(
                  func=Attribute(
                    value=Name(
                      id='outputs',
                      ctx=Load()),
                    attr='max',
                    ctx=Load()),
                  args=[Num(n=1)],
                  keywords=[])),
              AugAssign(
                target=Name(
                  id='total',
                  ctx=Store()),
                op=Add(),
                value=Call(
                  func=Attribute(
                    value=Name(
                      id='targets',
                      ctx=Load()),
                    attr='size',
                    ctx=Load()),
                  args=[Num(n=0)],
                  keywords=[])),
              AugAssign(
                target=Name(
                  id='correct',
                  ctx=Store()),
                op=Add(),
                value=Call(
                  func=Attribute(
                    value=Call(
                      func=Attribute(
                        value=Call(
                          func=Attribute(
                            value=Name(
                              id='predicted',
                              ctx=Load()),
                            attr='eq',
                            ctx=Load()),
                          args=[Name(
                            id='targets',
                            ctx=Load())],
                          keywords=[]),
                        attr='sum',
                        ctx=Load()),
                      args=[],
                      keywords=[]),
                    attr='item',
                    ctx=Load()),
                  args=[],
                  keywords=[]))],
            orelse=[]),
          Assign(
            targets=[Name(
              id='final_loss',
              ctx=Store())],
            value=Call(
              func=Name(
                id='round',
                ctx=Load()),
              args=[
                BinOp(
                  left=Name(
                    id='test_loss',
                    ctx=Load()),
                  op=Div(),
                  right=BinOp(
                    left=Name(
                      id='batch_idx',
                      ctx=Load()),
                    op=Add(),
                    right=Num(n=1))),
                Num(n=2)],
              keywords=[])),
          Assign(
            targets=[Name(
              id='final_acc',
              ctx=Store())],
            value=Call(
              func=Name(
                id='round',
                ctx=Load()),
              args=[
                BinOp(
                  left=BinOp(
                    left=Num(n=100),
                    op=Mult(),
                    right=Name(
                      id='correct',
                      ctx=Load())),
                  op=Div(),
                  right=Name(
                    id='total',
                    ctx=Load())),
                Num(n=2)],
              keywords=[])),
          Return(value=Tuple(
            elts=[
              Name(
                id='final_loss',
                ctx=Load()),
              Name(
                id='final_acc',
                ctx=Load())],
            ctx=Load()))])],
    decorator_list=[],
    returns=None),
  Assign(
    targets=[Name(
      id='train_losses',
      ctx=Store())],
    value=List(
      elts=[],
      ctx=Load())),
  Assign(
    targets=[Name(
      id='train_accs',
      ctx=Store())],
    value=List(
      elts=[],
      ctx=Load())),
  Assign(
    targets=[Name(
      id='test_losses',
      ctx=Store())],
    value=List(
      elts=[],
      ctx=Load())),
  Assign(
    targets=[Name(
      id='test_accs',
      ctx=Store())],
    value=List(
      elts=[],
      ctx=Load())),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[Str(s='Start training...')],
    keywords=[])),
  For(
    target=Name(
      id='epoch',
      ctx=Store()),
    iter=Call(
      func=Name(
        id='range',
        ctx=Load()),
      args=[Name(
        id='epochs',
        ctx=Load())],
      keywords=[]),
    body=[
      Assign(
        targets=[Tuple(
          elts=[
            Name(
              id='tr_l',
              ctx=Store()),
            Name(
              id='tr_a',
              ctx=Store())],
          ctx=Store())],
        value=Call(
          func=Name(
            id='train',
            ctx=Load()),
          args=[Name(
            id='epoch',
            ctx=Load())],
          keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='train_losses',
            ctx=Load()),
          attr='append',
          ctx=Load()),
        args=[Name(
          id='tr_l',
          ctx=Load())],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='train_accs',
            ctx=Load()),
          attr='append',
          ctx=Load()),
        args=[Name(
          id='tr_a',
          ctx=Load())],
        keywords=[])),
      Assign(
        targets=[Tuple(
          elts=[
            Name(
              id='ts_l',
              ctx=Store()),
            Name(
              id='ts_a',
              ctx=Store())],
          ctx=Store())],
        value=Call(
          func=Name(
            id='test',
            ctx=Load()),
          args=[Name(
            id='epoch',
            ctx=Load())],
          keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='test_losses',
            ctx=Load()),
          attr='append',
          ctx=Load()),
        args=[Name(
          id='ts_l',
          ctx=Load())],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='test_accs',
            ctx=Load()),
          attr='append',
          ctx=Load()),
        args=[Name(
          id='ts_a',
          ctx=Load())],
        keywords=[])),
      Expr(value=Call(
        func=Name(
          id='print',
          ctx=Load()),
        args=[JoinedStr(values=[
          Str(s='Epoch '),
          FormattedValue(
            value=Name(
              id='epoch',
              ctx=Load()),
            conversion=-1,
            format_spec=None),
          Str(s=': train_loss: '),
          FormattedValue(
            value=Name(
              id='tr_l',
              ctx=Load()),
            conversion=-1,
            format_spec=None),
          Str(s=' train_acc: '),
          FormattedValue(
            value=Name(
              id='tr_a',
              ctx=Load()),
            conversion=-1,
            format_spec=None),
          Str(s=' | test_loss: '),
          FormattedValue(
            value=Name(
              id='ts_l',
              ctx=Load()),
            conversion=-1,
            format_spec=None),
          Str(s=' test_acc: '),
          FormattedValue(
            value=Name(
              id='ts_a',
              ctx=Load()),
            conversion=-1,
            format_spec=None)])],
        keywords=[]))],
    orelse=[])])