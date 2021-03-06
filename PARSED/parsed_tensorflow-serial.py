Module(body=[
  Import(names=[alias(
    name='os',
    asname=None)]),
  Import(names=[alias(
    name='math',
    asname=None)]),
  Import(names=[alias(
    name='tensorflow',
    asname='tf')]),
  ImportFrom(
    module='tensorflow.keras.datasets',
    names=[alias(
      name='cifar10',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.preprocessing.image',
    names=[alias(
      name='ImageDataGenerator',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.models',
    names=[alias(
      name='Sequential',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.layers',
    names=[
      alias(
        name='Dense',
        asname=None),
      alias(
        name='Dropout',
        asname=None),
      alias(
        name='Activation',
        asname=None),
      alias(
        name='Flatten',
        asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.layers',
    names=[
      alias(
        name='Conv2D',
        asname=None),
      alias(
        name='MaxPooling2D',
        asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.callbacks',
    names=[alias(
      name='ModelCheckpoint',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.applications',
    names=[alias(
      name='ResNet50',
      asname=None)],
    level=0),
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
  Expr(value=Call(
    func=Attribute(
      value=Attribute(
        value=Name(
          id='tf',
          ctx=Load()),
        attr='random',
        ctx=Load()),
      attr='set_seed',
      ctx=Load()),
    args=[Num(n=1)],
    keywords=[])),
  Assign(
    targets=[Tuple(
      elts=[
        Tuple(
          elts=[
            Name(
              id='x_train',
              ctx=Store()),
            Name(
              id='y_train',
              ctx=Store())],
          ctx=Store()),
        Tuple(
          elts=[
            Name(
              id='x_test',
              ctx=Store()),
            Name(
              id='y_test',
              ctx=Store())],
          ctx=Store())],
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='cifar10',
          ctx=Load()),
        attr='load_data',
        ctx=Load()),
      args=[],
      keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='x_train shape:'),
      Attribute(
        value=Name(
          id='x_train',
          ctx=Load()),
        attr='shape',
        ctx=Load())],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Subscript(
        value=Attribute(
          value=Name(
            id='x_train',
            ctx=Load()),
          attr='shape',
          ctx=Load()),
        slice=Index(value=Num(n=0)),
        ctx=Load()),
      Str(s='train samples')],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Subscript(
        value=Attribute(
          value=Name(
            id='x_test',
            ctx=Load()),
          attr='shape',
          ctx=Load()),
        slice=Index(value=Num(n=0)),
        ctx=Load()),
      Str(s='test samples')],
    keywords=[])),
  Assign(
    targets=[Name(
      id='y_train',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Attribute(
            value=Name(
              id='tf',
              ctx=Load()),
            attr='keras',
            ctx=Load()),
          attr='utils',
          ctx=Load()),
        attr='to_categorical',
        ctx=Load()),
      args=[
        Name(
          id='y_train',
          ctx=Load()),
        Name(
          id='num_classes',
          ctx=Load())],
      keywords=[])),
  Assign(
    targets=[Name(
      id='y_test',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Attribute(
          value=Attribute(
            value=Name(
              id='tf',
              ctx=Load()),
            attr='keras',
            ctx=Load()),
          attr='utils',
          ctx=Load()),
        attr='to_categorical',
        ctx=Load()),
      args=[
        Name(
          id='y_test',
          ctx=Load()),
        Name(
          id='num_classes',
          ctx=Load())],
      keywords=[])),
  FunctionDef(
    name='define_model',
    args=arguments(
      args=[],
      vararg=None,
      kwonlyargs=[],
      kw_defaults=[],
      kwarg=None,
      defaults=[]),
    body=[
      Assign(
        targets=[Name(
          id='model',
          ctx=Store())],
        value=Call(
          func=Name(
            id='Sequential',
            ctx=Load()),
          args=[],
          keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Conv2D',
            ctx=Load()),
          args=[
            Num(n=64),
            Tuple(
              elts=[
                Num(n=3),
                Num(n=3)],
              ctx=Load())],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform')),
            keyword(
              arg='padding',
              value=Str(s='same')),
            keyword(
              arg='input_shape',
              value=Tuple(
                elts=[
                  Num(n=32),
                  Num(n=32),
                  Num(n=3)],
                ctx=Load()))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Conv2D',
            ctx=Load()),
          args=[
            Num(n=64),
            Tuple(
              elts=[
                Num(n=3),
                Num(n=3)],
              ctx=Load())],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform')),
            keyword(
              arg='padding',
              value=Str(s='same'))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='MaxPooling2D',
            ctx=Load()),
          args=[Tuple(
            elts=[
              Num(n=2),
              Num(n=2)],
            ctx=Load())],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Conv2D',
            ctx=Load()),
          args=[
            Num(n=1024),
            Tuple(
              elts=[
                Num(n=3),
                Num(n=3)],
              ctx=Load())],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform')),
            keyword(
              arg='padding',
              value=Str(s='same'))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Conv2D',
            ctx=Load()),
          args=[
            Num(n=1024),
            Tuple(
              elts=[
                Num(n=3),
                Num(n=3)],
              ctx=Load())],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform')),
            keyword(
              arg='padding',
              value=Str(s='same'))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='MaxPooling2D',
            ctx=Load()),
          args=[Tuple(
            elts=[
              Num(n=2),
              Num(n=2)],
            ctx=Load())],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Conv2D',
            ctx=Load()),
          args=[
            Num(n=1024),
            Tuple(
              elts=[
                Num(n=3),
                Num(n=3)],
              ctx=Load())],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform')),
            keyword(
              arg='padding',
              value=Str(s='same'))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Conv2D',
            ctx=Load()),
          args=[
            Num(n=1024),
            Tuple(
              elts=[
                Num(n=3),
                Num(n=3)],
              ctx=Load())],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform')),
            keyword(
              arg='padding',
              value=Str(s='same'))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='MaxPooling2D',
            ctx=Load()),
          args=[Tuple(
            elts=[
              Num(n=2),
              Num(n=2)],
            ctx=Load())],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Flatten',
            ctx=Load()),
          args=[],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Dense',
            ctx=Load()),
          args=[Num(n=128)],
          keywords=[
            keyword(
              arg='activation',
              value=Str(s='relu')),
            keyword(
              arg='kernel_initializer',
              value=Str(s='he_uniform'))])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Dropout',
            ctx=Load()),
          args=[Num(n=0.2)],
          keywords=[])],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='add',
          ctx=Load()),
        args=[Call(
          func=Name(
            id='Dense',
            ctx=Load()),
          args=[Num(n=10)],
          keywords=[keyword(
            arg='activation',
            value=Str(s='softmax'))])],
        keywords=[])),
      Assign(
        targets=[Name(
          id='opt',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Attribute(
              value=Attribute(
                value=Name(
                  id='tf',
                  ctx=Load()),
                attr='keras',
                ctx=Load()),
              attr='optimizers',
              ctx=Load()),
            attr='SGD',
            ctx=Load()),
          args=[],
          keywords=[
            keyword(
              arg='learning_rate',
              value=Num(n=0.001)),
            keyword(
              arg='momentum',
              value=Num(n=0.9)),
            keyword(
              arg='nesterov',
              value=NameConstant(value=False))])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='model',
            ctx=Load()),
          attr='compile',
          ctx=Load()),
        args=[],
        keywords=[
          keyword(
            arg='optimizer',
            value=Name(
              id='opt',
              ctx=Load())),
          keyword(
            arg='loss',
            value=Str(s='categorical_crossentropy')),
          keyword(
            arg='metrics',
            value=List(
              elts=[Str(s='accuracy')],
              ctx=Load()))])),
      Return(value=Name(
        id='model',
        ctx=Load()))],
    decorator_list=[],
    returns=None),
  Assign(
    targets=[Name(
      id='model',
      ctx=Store())],
    value=Call(
      func=Name(
        id='define_model',
        ctx=Load()),
      args=[],
      keywords=[])),
  Assign(
    targets=[Name(
      id='x_train',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='x_train',
          ctx=Load()),
        attr='astype',
        ctx=Load()),
      args=[Str(s='float32')],
      keywords=[])),
  Assign(
    targets=[Name(
      id='x_test',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='x_test',
          ctx=Load()),
        attr='astype',
        ctx=Load()),
      args=[Str(s='float32')],
      keywords=[])),
  AugAssign(
    target=Name(
      id='x_train',
      ctx=Store()),
    op=Div(),
    value=Num(n=255)),
  AugAssign(
    target=Name(
      id='x_test',
      ctx=Store()),
    op=Div(),
    value=Num(n=255)),
  Assign(
    targets=[Name(
      id='history',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='model',
          ctx=Load()),
        attr='fit',
        ctx=Load()),
      args=[
        Name(
          id='x_train',
          ctx=Load()),
        Name(
          id='y_train',
          ctx=Load())],
      keywords=[
        keyword(
          arg='batch_size',
          value=Name(
            id='batch_size',
            ctx=Load())),
        keyword(
          arg='epochs',
          value=Name(
            id='epochs',
            ctx=Load())),
        keyword(
          arg='validation_data',
          value=Tuple(
            elts=[
              Name(
                id='x_test',
                ctx=Load()),
              Name(
                id='y_test',
                ctx=Load())],
            ctx=Load()))])),
  Assign(
    targets=[Name(
      id='scores',
      ctx=Store())],
    value=Call(
      func=Attribute(
        value=Name(
          id='model',
          ctx=Load()),
        attr='evaluate',
        ctx=Load()),
      args=[
        Name(
          id='x_test',
          ctx=Load()),
        Name(
          id='y_test',
          ctx=Load())],
      keywords=[keyword(
        arg='verbose',
        value=Num(n=1))])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='Test loss:'),
      Subscript(
        value=Name(
          id='scores',
          ctx=Load()),
        slice=Index(value=Num(n=0)),
        ctx=Load())],
    keywords=[])),
  Expr(value=Call(
    func=Name(
      id='print',
      ctx=Load()),
    args=[
      Str(s='Test accuracy:'),
      Subscript(
        value=Name(
          id='scores',
          ctx=Load()),
        slice=Index(value=Num(n=1)),
        ctx=Load())],
    keywords=[]))])