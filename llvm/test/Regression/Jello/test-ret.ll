; test return instructions

void %test() { ret void }
sbyte %test() { ret sbyte 1 }
ubyte %test() { ret ubyte 1 }
short %test() { ret short -1 }
ushort %test() { ret ushort 65535 }
int  %test() { ret int -1 }
uint %test() { ret uint 4 }
;long %test() { ret void }
;ulong %test() { ret void }
;float %test() { ret float 1.0 }
;double %test() { ret double 2.0 }
