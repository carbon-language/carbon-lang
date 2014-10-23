$c1 = comdat largest

@some_name = private unnamed_addr constant i32 42, comdat $c1
@c1 = alias i32* @some_name
