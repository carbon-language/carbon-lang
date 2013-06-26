_CLC_OVERLOAD _CLC_DEF GENTYPE max(GENTYPE a, GENTYPE b) {
  return (a > b ? a : b);
}

#ifndef SCALAR
_CLC_OVERLOAD _CLC_DEF GENTYPE max(GENTYPE a, SCALAR_GENTYPE b) {
  return (a > (GENTYPE)b ? a : (GENTYPE)b);
}
#endif