; We were accidentally inverting the signedness of right shifts.  Whoops.

int %main() {
  %X = shr int -1, ubyte 16
  %Y = shr int %X, ubyte 16
  %Z = add int %Y, 1
  ret int %Z
}
