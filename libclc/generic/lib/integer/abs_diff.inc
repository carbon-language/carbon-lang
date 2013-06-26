_CLC_OVERLOAD _CLC_DEF UGENTYPE abs_diff(GENTYPE x, GENTYPE y) {
  return __builtin_astype((GENTYPE)(x > y ? x-y : y-x), UGENTYPE);
}
