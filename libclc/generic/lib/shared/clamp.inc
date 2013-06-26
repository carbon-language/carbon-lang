_CLC_OVERLOAD _CLC_DEF GENTYPE clamp(GENTYPE x, GENTYPE y, GENTYPE z) {
  return (x > z ? z : (x < y ? y : x));
}
