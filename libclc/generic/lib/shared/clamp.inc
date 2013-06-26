_CLC_OVERLOAD _CLC_DEF GENTYPE clamp(GENTYPE x, GENTYPE y, GENTYPE z) {
  return (x > z ? z : (x < y ? y : x));
}

#ifndef SCALAR
_CLC_OVERLOAD _CLC_DEF GENTYPE clamp(GENTYPE x, SCALAR_GENTYPE y, SCALAR_GENTYPE z) {
  return (x > (GENTYPE)z ? (GENTYPE)z : (x < (GENTYPE)y ? (GENTYPE)y : x));
}
#endif