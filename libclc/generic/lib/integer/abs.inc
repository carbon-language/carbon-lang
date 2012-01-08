_CLC_OVERLOAD _CLC_DEF UGENTYPE abs(GENTYPE x) {
  return __builtin_astype((GENTYPE)(x > (GENTYPE)(0) ? x : -x), UGENTYPE);
}
