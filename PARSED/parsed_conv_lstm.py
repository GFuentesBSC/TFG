Module(body=[
  Expr(value=Str(s='\n#This script demonstrates the use of a convolutional LSTM network.\nThis network is used to predict the next frame of an artificially\ngenerated movie which contains moving squares.\n')),
  ImportFrom(
    module='tensorflow.keras.models',
    names=[alias(
      name='Sequential',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.layers',
    names=[alias(
      name='Conv3D',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.layers',
    names=[alias(
      name='ConvLSTM2D',
      asname=None)],
    level=0),
  ImportFrom(
    module='tensorflow.keras.layers',
    names=[alias(
      name='BatchNormalization',
      asname=None)],
    level=0),
  Import(names=[alias(
    name='numpy',
    asname='np')]),
  Import(names=[alias(
    name='pylab',
    asname='plt')]),
  Assign(
    targets=[Name(
      id='seq',
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
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='ConvLSTM2D',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='filters',
          value=Num(n=40)),
        keyword(
          arg='kernel_size',
          value=Tuple(
            elts=[
              Num(n=3),
              Num(n=3)],
            ctx=Load())),
        keyword(
          arg='input_shape',
          value=Tuple(
            elts=[
              NameConstant(value=None),
              Num(n=40),
              Num(n=40),
              Num(n=1)],
            ctx=Load())),
        keyword(
          arg='padding',
          value=Str(s='same')),
        keyword(
          arg='return_sequences',
          value=NameConstant(value=True))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='BatchNormalization',
        ctx=Load()),
      args=[],
      keywords=[])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='ConvLSTM2D',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='filters',
          value=Num(n=40)),
        keyword(
          arg='kernel_size',
          value=Tuple(
            elts=[
              Num(n=3),
              Num(n=3)],
            ctx=Load())),
        keyword(
          arg='padding',
          value=Str(s='same')),
        keyword(
          arg='return_sequences',
          value=NameConstant(value=True))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='BatchNormalization',
        ctx=Load()),
      args=[],
      keywords=[])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='ConvLSTM2D',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='filters',
          value=Num(n=40)),
        keyword(
          arg='kernel_size',
          value=Tuple(
            elts=[
              Num(n=3),
              Num(n=3)],
            ctx=Load())),
        keyword(
          arg='padding',
          value=Str(s='same')),
        keyword(
          arg='return_sequences',
          value=NameConstant(value=True))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='BatchNormalization',
        ctx=Load()),
      args=[],
      keywords=[])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='ConvLSTM2D',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='filters',
          value=Num(n=40)),
        keyword(
          arg='kernel_size',
          value=Tuple(
            elts=[
              Num(n=3),
              Num(n=3)],
            ctx=Load())),
        keyword(
          arg='padding',
          value=Str(s='same')),
        keyword(
          arg='return_sequences',
          value=NameConstant(value=True))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='BatchNormalization',
        ctx=Load()),
      args=[],
      keywords=[])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='add',
      ctx=Load()),
    args=[Call(
      func=Name(
        id='Conv3D',
        ctx=Load()),
      args=[],
      keywords=[
        keyword(
          arg='filters',
          value=Num(n=1)),
        keyword(
          arg='kernel_size',
          value=Tuple(
            elts=[
              Num(n=3),
              Num(n=3),
              Num(n=3)],
            ctx=Load())),
        keyword(
          arg='activation',
          value=Str(s='sigmoid')),
        keyword(
          arg='padding',
          value=Str(s='same')),
        keyword(
          arg='data_format',
          value=Str(s='channels_last'))])],
    keywords=[])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='compile',
      ctx=Load()),
    args=[],
    keywords=[
      keyword(
        arg='loss',
        value=Str(s='binary_crossentropy')),
      keyword(
        arg='optimizer',
        value=Str(s='adadelta'))])),
  FunctionDef(
    name='generate_movies',
    args=arguments(
      args=[
        arg(
          arg='n_samples',
          annotation=None),
        arg(
          arg='n_frames',
          annotation=None)],
      vararg=None,
      kwonlyargs=[],
      kw_defaults=[],
      kwarg=None,
      defaults=[
        Num(n=1200),
        Num(n=15)]),
    body=[
      Assign(
        targets=[Name(
          id='row',
          ctx=Store())],
        value=Num(n=80)),
      Assign(
        targets=[Name(
          id='col',
          ctx=Store())],
        value=Num(n=80)),
      Assign(
        targets=[Name(
          id='noisy_movies',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='np',
              ctx=Load()),
            attr='zeros',
            ctx=Load()),
          args=[Tuple(
            elts=[
              Name(
                id='n_samples',
                ctx=Load()),
              Name(
                id='n_frames',
                ctx=Load()),
              Name(
                id='row',
                ctx=Load()),
              Name(
                id='col',
                ctx=Load()),
              Num(n=1)],
            ctx=Load())],
          keywords=[keyword(
            arg='dtype',
            value=Attribute(
              value=Name(
                id='np',
                ctx=Load()),
              attr='float',
              ctx=Load()))])),
      Assign(
        targets=[Name(
          id='shifted_movies',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='np',
              ctx=Load()),
            attr='zeros',
            ctx=Load()),
          args=[Tuple(
            elts=[
              Name(
                id='n_samples',
                ctx=Load()),
              Name(
                id='n_frames',
                ctx=Load()),
              Name(
                id='row',
                ctx=Load()),
              Name(
                id='col',
                ctx=Load()),
              Num(n=1)],
            ctx=Load())],
          keywords=[keyword(
            arg='dtype',
            value=Attribute(
              value=Name(
                id='np',
                ctx=Load()),
              attr='float',
              ctx=Load()))])),
      For(
        target=Name(
          id='i',
          ctx=Store()),
        iter=Call(
          func=Name(
            id='range',
            ctx=Load()),
          args=[Name(
            id='n_samples',
            ctx=Load())],
          keywords=[]),
        body=[
          Assign(
            targets=[Name(
              id='n',
              ctx=Store())],
            value=Call(
              func=Attribute(
                value=Attribute(
                  value=Name(
                    id='np',
                    ctx=Load()),
                  attr='random',
                  ctx=Load()),
                attr='randint',
                ctx=Load()),
              args=[
                Num(n=3),
                Num(n=8)],
              keywords=[])),
          For(
            target=Name(
              id='j',
              ctx=Store()),
            iter=Call(
              func=Name(
                id='range',
                ctx=Load()),
              args=[Name(
                id='n',
                ctx=Load())],
              keywords=[]),
            body=[
              Assign(
                targets=[Name(
                  id='xstart',
                  ctx=Store())],
                value=Call(
                  func=Attribute(
                    value=Attribute(
                      value=Name(
                        id='np',
                        ctx=Load()),
                      attr='random',
                      ctx=Load()),
                    attr='randint',
                    ctx=Load()),
                  args=[
                    Num(n=20),
                    Num(n=60)],
                  keywords=[])),
              Assign(
                targets=[Name(
                  id='ystart',
                  ctx=Store())],
                value=Call(
                  func=Attribute(
                    value=Attribute(
                      value=Name(
                        id='np',
                        ctx=Load()),
                      attr='random',
                      ctx=Load()),
                    attr='randint',
                    ctx=Load()),
                  args=[
                    Num(n=20),
                    Num(n=60)],
                  keywords=[])),
              Assign(
                targets=[Name(
                  id='directionx',
                  ctx=Store())],
                value=BinOp(
                  left=Call(
                    func=Attribute(
                      value=Attribute(
                        value=Name(
                          id='np',
                          ctx=Load()),
                        attr='random',
                        ctx=Load()),
                      attr='randint',
                      ctx=Load()),
                    args=[
                      Num(n=0),
                      Num(n=3)],
                    keywords=[]),
                  op=Sub(),
                  right=Num(n=1))),
              Assign(
                targets=[Name(
                  id='directiony',
                  ctx=Store())],
                value=BinOp(
                  left=Call(
                    func=Attribute(
                      value=Attribute(
                        value=Name(
                          id='np',
                          ctx=Load()),
                        attr='random',
                        ctx=Load()),
                      attr='randint',
                      ctx=Load()),
                    args=[
                      Num(n=0),
                      Num(n=3)],
                    keywords=[]),
                  op=Sub(),
                  right=Num(n=1))),
              Assign(
                targets=[Name(
                  id='w',
                  ctx=Store())],
                value=Call(
                  func=Attribute(
                    value=Attribute(
                      value=Name(
                        id='np',
                        ctx=Load()),
                      attr='random',
                      ctx=Load()),
                    attr='randint',
                    ctx=Load()),
                  args=[
                    Num(n=2),
                    Num(n=4)],
                  keywords=[])),
              For(
                target=Name(
                  id='t',
                  ctx=Store()),
                iter=Call(
                  func=Name(
                    id='range',
                    ctx=Load()),
                  args=[Name(
                    id='n_frames',
                    ctx=Load())],
                  keywords=[]),
                body=[
                  Assign(
                    targets=[Name(
                      id='x_shift',
                      ctx=Store())],
                    value=BinOp(
                      left=Name(
                        id='xstart',
                        ctx=Load()),
                      op=Add(),
                      right=BinOp(
                        left=Name(
                          id='directionx',
                          ctx=Load()),
                        op=Mult(),
                        right=Name(
                          id='t',
                          ctx=Load())))),
                  Assign(
                    targets=[Name(
                      id='y_shift',
                      ctx=Store())],
                    value=BinOp(
                      left=Name(
                        id='ystart',
                        ctx=Load()),
                      op=Add(),
                      right=BinOp(
                        left=Name(
                          id='directiony',
                          ctx=Load()),
                        op=Mult(),
                        right=Name(
                          id='t',
                          ctx=Load())))),
                  AugAssign(
                    target=Subscript(
                      value=Name(
                        id='noisy_movies',
                        ctx=Load()),
                      slice=ExtSlice(dims=[
                        Index(value=Name(
                          id='i',
                          ctx=Load())),
                        Index(value=Name(
                          id='t',
                          ctx=Load())),
                        Slice(
                          lower=BinOp(
                            left=Name(
                              id='x_shift',
                              ctx=Load()),
                            op=Sub(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          upper=BinOp(
                            left=Name(
                              id='x_shift',
                              ctx=Load()),
                            op=Add(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          step=None),
                        Slice(
                          lower=BinOp(
                            left=Name(
                              id='y_shift',
                              ctx=Load()),
                            op=Sub(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          upper=BinOp(
                            left=Name(
                              id='y_shift',
                              ctx=Load()),
                            op=Add(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          step=None),
                        Index(value=Num(n=0))]),
                      ctx=Store()),
                    op=Add(),
                    value=Num(n=1)),
                  If(
                    test=Call(
                      func=Attribute(
                        value=Attribute(
                          value=Name(
                            id='np',
                            ctx=Load()),
                          attr='random',
                          ctx=Load()),
                        attr='randint',
                        ctx=Load()),
                      args=[
                        Num(n=0),
                        Num(n=2)],
                      keywords=[]),
                    body=[
                      Assign(
                        targets=[Name(
                          id='noise_f',
                          ctx=Store())],
                        value=BinOp(
                          left=UnaryOp(
                            op=USub(),
                            operand=Num(n=1)),
                          op=Pow(),
                          right=Call(
                            func=Attribute(
                              value=Attribute(
                                value=Name(
                                  id='np',
                                  ctx=Load()),
                                attr='random',
                                ctx=Load()),
                              attr='randint',
                              ctx=Load()),
                            args=[
                              Num(n=0),
                              Num(n=2)],
                            keywords=[]))),
                      AugAssign(
                        target=Subscript(
                          value=Name(
                            id='noisy_movies',
                            ctx=Load()),
                          slice=ExtSlice(dims=[
                            Index(value=Name(
                              id='i',
                              ctx=Load())),
                            Index(value=Name(
                              id='t',
                              ctx=Load())),
                            Slice(
                              lower=BinOp(
                                left=BinOp(
                                  left=Name(
                                    id='x_shift',
                                    ctx=Load()),
                                  op=Sub(),
                                  right=Name(
                                    id='w',
                                    ctx=Load())),
                                op=Sub(),
                                right=Num(n=1)),
                              upper=BinOp(
                                left=BinOp(
                                  left=Name(
                                    id='x_shift',
                                    ctx=Load()),
                                  op=Add(),
                                  right=Name(
                                    id='w',
                                    ctx=Load())),
                                op=Add(),
                                right=Num(n=1)),
                              step=None),
                            Slice(
                              lower=BinOp(
                                left=BinOp(
                                  left=Name(
                                    id='y_shift',
                                    ctx=Load()),
                                  op=Sub(),
                                  right=Name(
                                    id='w',
                                    ctx=Load())),
                                op=Sub(),
                                right=Num(n=1)),
                              upper=BinOp(
                                left=BinOp(
                                  left=Name(
                                    id='y_shift',
                                    ctx=Load()),
                                  op=Add(),
                                  right=Name(
                                    id='w',
                                    ctx=Load())),
                                op=Add(),
                                right=Num(n=1)),
                              step=None),
                            Index(value=Num(n=0))]),
                          ctx=Store()),
                        op=Add(),
                        value=BinOp(
                          left=Name(
                            id='noise_f',
                            ctx=Load()),
                          op=Mult(),
                          right=Num(n=0.1)))],
                    orelse=[]),
                  Assign(
                    targets=[Name(
                      id='x_shift',
                      ctx=Store())],
                    value=BinOp(
                      left=Name(
                        id='xstart',
                        ctx=Load()),
                      op=Add(),
                      right=BinOp(
                        left=Name(
                          id='directionx',
                          ctx=Load()),
                        op=Mult(),
                        right=BinOp(
                          left=Name(
                            id='t',
                            ctx=Load()),
                          op=Add(),
                          right=Num(n=1))))),
                  Assign(
                    targets=[Name(
                      id='y_shift',
                      ctx=Store())],
                    value=BinOp(
                      left=Name(
                        id='ystart',
                        ctx=Load()),
                      op=Add(),
                      right=BinOp(
                        left=Name(
                          id='directiony',
                          ctx=Load()),
                        op=Mult(),
                        right=BinOp(
                          left=Name(
                            id='t',
                            ctx=Load()),
                          op=Add(),
                          right=Num(n=1))))),
                  AugAssign(
                    target=Subscript(
                      value=Name(
                        id='shifted_movies',
                        ctx=Load()),
                      slice=ExtSlice(dims=[
                        Index(value=Name(
                          id='i',
                          ctx=Load())),
                        Index(value=Name(
                          id='t',
                          ctx=Load())),
                        Slice(
                          lower=BinOp(
                            left=Name(
                              id='x_shift',
                              ctx=Load()),
                            op=Sub(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          upper=BinOp(
                            left=Name(
                              id='x_shift',
                              ctx=Load()),
                            op=Add(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          step=None),
                        Slice(
                          lower=BinOp(
                            left=Name(
                              id='y_shift',
                              ctx=Load()),
                            op=Sub(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          upper=BinOp(
                            left=Name(
                              id='y_shift',
                              ctx=Load()),
                            op=Add(),
                            right=Name(
                              id='w',
                              ctx=Load())),
                          step=None),
                        Index(value=Num(n=0))]),
                      ctx=Store()),
                    op=Add(),
                    value=Num(n=1))],
                orelse=[])],
            orelse=[])],
        orelse=[]),
      Assign(
        targets=[Name(
          id='noisy_movies',
          ctx=Store())],
        value=Subscript(
          value=Name(
            id='noisy_movies',
            ctx=Load()),
          slice=ExtSlice(dims=[
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=Num(n=20),
              upper=Num(n=60),
              step=None),
            Slice(
              lower=Num(n=20),
              upper=Num(n=60),
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None)]),
          ctx=Load())),
      Assign(
        targets=[Name(
          id='shifted_movies',
          ctx=Store())],
        value=Subscript(
          value=Name(
            id='shifted_movies',
            ctx=Load()),
          slice=ExtSlice(dims=[
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=Num(n=20),
              upper=Num(n=60),
              step=None),
            Slice(
              lower=Num(n=20),
              upper=Num(n=60),
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None)]),
          ctx=Load())),
      Assign(
        targets=[Subscript(
          value=Name(
            id='noisy_movies',
            ctx=Load()),
          slice=Index(value=Compare(
            left=Name(
              id='noisy_movies',
              ctx=Load()),
            ops=[GtE()],
            comparators=[Num(n=1)])),
          ctx=Store())],
        value=Num(n=1)),
      Assign(
        targets=[Subscript(
          value=Name(
            id='shifted_movies',
            ctx=Load()),
          slice=Index(value=Compare(
            left=Name(
              id='shifted_movies',
              ctx=Load()),
            ops=[GtE()],
            comparators=[Num(n=1)])),
          ctx=Store())],
        value=Num(n=1)),
      Return(value=Tuple(
        elts=[
          Name(
            id='noisy_movies',
            ctx=Load()),
          Name(
            id='shifted_movies',
            ctx=Load())],
        ctx=Load()))],
    decorator_list=[],
    returns=None),
  Assign(
    targets=[Tuple(
      elts=[
        Name(
          id='noisy_movies',
          ctx=Store()),
        Name(
          id='shifted_movies',
          ctx=Store())],
      ctx=Store())],
    value=Call(
      func=Name(
        id='generate_movies',
        ctx=Load()),
      args=[],
      keywords=[keyword(
        arg='n_samples',
        value=Num(n=1200))])),
  Expr(value=Call(
    func=Attribute(
      value=Name(
        id='seq',
        ctx=Load()),
      attr='fit',
      ctx=Load()),
    args=[
      Subscript(
        value=Name(
          id='noisy_movies',
          ctx=Load()),
        slice=Slice(
          lower=None,
          upper=Num(n=1000),
          step=None),
        ctx=Load()),
      Subscript(
        value=Name(
          id='shifted_movies',
          ctx=Load()),
        slice=Slice(
          lower=None,
          upper=Num(n=1000),
          step=None),
        ctx=Load())],
    keywords=[
      keyword(
        arg='batch_size',
        value=Num(n=10)),
      keyword(
        arg='epochs',
        value=Num(n=300)),
      keyword(
        arg='validation_split',
        value=Num(n=0.05))])),
  Assign(
    targets=[Name(
      id='which',
      ctx=Store())],
    value=Num(n=1004)),
  Assign(
    targets=[Name(
      id='track',
      ctx=Store())],
    value=Subscript(
      value=Subscript(
        value=Name(
          id='noisy_movies',
          ctx=Load()),
        slice=Index(value=Name(
          id='which',
          ctx=Load())),
        ctx=Load()),
      slice=ExtSlice(dims=[
        Slice(
          lower=None,
          upper=Num(n=7),
          step=None),
        Slice(
          lower=None,
          upper=None,
          step=None),
        Slice(
          lower=None,
          upper=None,
          step=None),
        Slice(
          lower=None,
          upper=None,
          step=None)]),
      ctx=Load())),
  For(
    target=Name(
      id='j',
      ctx=Store()),
    iter=Call(
      func=Name(
        id='range',
        ctx=Load()),
      args=[Num(n=16)],
      keywords=[]),
    body=[
      Assign(
        targets=[Name(
          id='new_pos',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='seq',
              ctx=Load()),
            attr='predict',
            ctx=Load()),
          args=[Subscript(
            value=Name(
              id='track',
              ctx=Load()),
            slice=ExtSlice(dims=[
              Index(value=Attribute(
                value=Name(
                  id='np',
                  ctx=Load()),
                attr='newaxis',
                ctx=Load())),
              Slice(
                lower=None,
                upper=None,
                step=None),
              Slice(
                lower=None,
                upper=None,
                step=None),
              Slice(
                lower=None,
                upper=None,
                step=None),
              Slice(
                lower=None,
                upper=None,
                step=None)]),
            ctx=Load())],
          keywords=[])),
      Assign(
        targets=[Name(
          id='new',
          ctx=Store())],
        value=Subscript(
          value=Name(
            id='new_pos',
            ctx=Load()),
          slice=ExtSlice(dims=[
            Slice(
              lower=None,
              upper=None,
              step=None),
            Index(value=UnaryOp(
              op=USub(),
              operand=Num(n=1))),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None)]),
          ctx=Load())),
      Assign(
        targets=[Name(
          id='track',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='np',
              ctx=Load()),
            attr='concatenate',
            ctx=Load()),
          args=[Tuple(
            elts=[
              Name(
                id='track',
                ctx=Load()),
              Name(
                id='new',
                ctx=Load())],
            ctx=Load())],
          keywords=[keyword(
            arg='axis',
            value=Num(n=0))]))],
    orelse=[]),
  Assign(
    targets=[Name(
      id='track2',
      ctx=Store())],
    value=Subscript(
      value=Subscript(
        value=Name(
          id='noisy_movies',
          ctx=Load()),
        slice=Index(value=Name(
          id='which',
          ctx=Load())),
        ctx=Load()),
      slice=ExtSlice(dims=[
        Slice(
          lower=None,
          upper=None,
          step=None),
        Slice(
          lower=None,
          upper=None,
          step=None),
        Slice(
          lower=None,
          upper=None,
          step=None),
        Slice(
          lower=None,
          upper=None,
          step=None)]),
      ctx=Load())),
  For(
    target=Name(
      id='i',
      ctx=Store()),
    iter=Call(
      func=Name(
        id='range',
        ctx=Load()),
      args=[Num(n=15)],
      keywords=[]),
    body=[
      Assign(
        targets=[Name(
          id='fig',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='plt',
              ctx=Load()),
            attr='figure',
            ctx=Load()),
          args=[],
          keywords=[keyword(
            arg='figsize',
            value=Tuple(
              elts=[
                Num(n=10),
                Num(n=5)],
              ctx=Load()))])),
      Assign(
        targets=[Name(
          id='ax',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='fig',
              ctx=Load()),
            attr='add_subplot',
            ctx=Load()),
          args=[Num(n=121)],
          keywords=[])),
      If(
        test=Compare(
          left=Name(
            id='i',
            ctx=Load()),
          ops=[GtE()],
          comparators=[Num(n=7)]),
        body=[Expr(value=Call(
          func=Attribute(
            value=Name(
              id='ax',
              ctx=Load()),
            attr='text',
            ctx=Load()),
          args=[
            Num(n=1),
            Num(n=3),
            Str(s='Predictions !')],
          keywords=[
            keyword(
              arg='fontsize',
              value=Num(n=20)),
            keyword(
              arg='color',
              value=Str(s='w'))]))],
        orelse=[Expr(value=Call(
          func=Attribute(
            value=Name(
              id='ax',
              ctx=Load()),
            attr='text',
            ctx=Load()),
          args=[
            Num(n=1),
            Num(n=3),
            Str(s='Initial trajectory')],
          keywords=[keyword(
            arg='fontsize',
            value=Num(n=20))]))]),
      Assign(
        targets=[Name(
          id='toplot',
          ctx=Store())],
        value=Subscript(
          value=Name(
            id='track',
            ctx=Load()),
          slice=ExtSlice(dims=[
            Index(value=Name(
              id='i',
              ctx=Load())),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Index(value=Num(n=0))]),
          ctx=Load())),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='plt',
            ctx=Load()),
          attr='imshow',
          ctx=Load()),
        args=[Name(
          id='toplot',
          ctx=Load())],
        keywords=[])),
      Assign(
        targets=[Name(
          id='ax',
          ctx=Store())],
        value=Call(
          func=Attribute(
            value=Name(
              id='fig',
              ctx=Load()),
            attr='add_subplot',
            ctx=Load()),
          args=[Num(n=122)],
          keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='plt',
            ctx=Load()),
          attr='text',
          ctx=Load()),
        args=[
          Num(n=1),
          Num(n=3),
          Str(s='Ground truth')],
        keywords=[keyword(
          arg='fontsize',
          value=Num(n=20))])),
      Assign(
        targets=[Name(
          id='toplot',
          ctx=Store())],
        value=Subscript(
          value=Name(
            id='track2',
            ctx=Load()),
          slice=ExtSlice(dims=[
            Index(value=Name(
              id='i',
              ctx=Load())),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Slice(
              lower=None,
              upper=None,
              step=None),
            Index(value=Num(n=0))]),
          ctx=Load())),
      If(
        test=Compare(
          left=Name(
            id='i',
            ctx=Load()),
          ops=[GtE()],
          comparators=[Num(n=2)]),
        body=[Assign(
          targets=[Name(
            id='toplot',
            ctx=Store())],
          value=Subscript(
            value=Subscript(
              value=Name(
                id='shifted_movies',
                ctx=Load()),
              slice=Index(value=Name(
                id='which',
                ctx=Load())),
              ctx=Load()),
            slice=ExtSlice(dims=[
              Index(value=BinOp(
                left=Name(
                  id='i',
                  ctx=Load()),
                op=Sub(),
                right=Num(n=1))),
              Slice(
                lower=None,
                upper=None,
                step=None),
              Slice(
                lower=None,
                upper=None,
                step=None),
              Index(value=Num(n=0))]),
            ctx=Load()))],
        orelse=[]),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='plt',
            ctx=Load()),
          attr='imshow',
          ctx=Load()),
        args=[Name(
          id='toplot',
          ctx=Load())],
        keywords=[])),
      Expr(value=Call(
        func=Attribute(
          value=Name(
            id='plt',
            ctx=Load()),
          attr='savefig',
          ctx=Load()),
        args=[BinOp(
          left=Str(s='%i_animate.png'),
          op=Mod(),
          right=BinOp(
            left=Name(
              id='i',
              ctx=Load()),
            op=Add(),
            right=Num(n=1)))],
        keywords=[]))],
    orelse=[])])