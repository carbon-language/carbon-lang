// RUN: tblgen %s 
// XFAIL: vg_leak
class x {
  string y = "missing terminating '\"' character";
}

