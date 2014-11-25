%u = type { i8 }

@g2 = global %u zeroinitializer
@a = weak alias %u* @g2
