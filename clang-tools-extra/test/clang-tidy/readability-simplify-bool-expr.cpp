// RUN: $(dirname %s)/check_clang_tidy.sh %s readability-simplify-boolean-expr %t
// REQUIRES: shell

bool a1 = false;

//=-=-=-=-=-=-= operator ==
bool aa = false == a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant boolean literal supplied to boolean operator [readability-simplify-boolean-expr]
// CHECK-FIXES: {{^bool aa = !a1;$}}
bool ab = true == a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool ab = a1;$}}
bool a2 = a1 == false;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a2 = !a1;$}}
bool a3 = a1 == true;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a3 = a1;$}}

//=-=-=-=-=-=-= operator !=
bool n1 = a1 != false;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool n1 = a1;$}}
bool n2 = a1 != true;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool n2 = !a1;$}}
bool n3 = false != a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool n3 = a1;$}}
bool n4 = true != a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool n4 = !a1;$}}

//=-=-=-=-=-=-= operator ||
bool a4 = a1 || false;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a4 = a1;$}}
bool a5 = a1 || true;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a5 = true;$}}
bool a6 = false || a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a6 = a1;$}}
bool a7 = true || a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a7 = true;$}}

//=-=-=-=-=-=-= operator &&
bool a8 = a1 && false;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a8 = false;$}}
bool a9 = a1 && true;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool a9 = a1;$}}
bool ac = false && a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool ac = false;$}}
bool ad = true && a1;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: {{.*}} to boolean operator
// CHECK-FIXES: {{^bool ad = a1;$}}

void if_with_bool_literal_condition() {
  int i = 0;
  if (false) {
    i = 1;
  } else {
    i = 2;
  }
  i = 3;
  // CHECK-MESSAGES: :[[@LINE-6]]:7: warning: {{.*}} in if statement condition
  // CHECK-FIXES:      {{^  int i = 0;$}}
  // CHECK-FIXES-NEXT: {{^  {$}}
  // CHECK-FIXES-NEXT: {{^    i = 2;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
  // CHECK-FIXES-NEXT: {{^  i = 3;$}}

  i = 4;
  if (true) {
    i = 5;
  } else {
    i = 6;
  }
  i = 7;
  // CHECK-MESSAGES: :[[@LINE-6]]:7: warning: {{.*}} in if statement condition
  // CHECK-FIXES:      {{^  i = 4;$}}
  // CHECK-FIXES-NEXT: {{^  {$}}
  // CHECK-FIXES-NEXT: {{^    i = 5;$}}
  // CHECK-FIXES-NEXT: {{^  }$}}
  // CHECK-FIXES-NEXT: {{^  i = 7;$}}

  i = 8;
  if (false) {
    i = 9;
  }
  i = 11;
  // CHECK-MESSAGES: :[[@LINE-4]]:7: warning: {{.*}} in if statement condition
  // CHECK-FIXES:      {{^  i = 8;$}}
  // CHECK-FIXES-NEXT: {{^  $}}
  // CHECK-FIXES-NEXT: {{^  i = 11;$}}
}

void operator_equals() {
  int i = 0;
  bool b1 = (i > 2);
  if (b1 == true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(b1\) {$}}
    i = 5;
  } else {
    i = 6;
  }
  bool b2 = (i > 4);
  if (b2 == false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!b2\) {$}}
    i = 7;
  } else {
    i = 9;
  }
  bool b3 = (i > 6);
  if (true == b3) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(b3\) {$}}
    i = 10;
  } else {
    i = 11;
  }
  bool b4 = (i > 8);
  if (false == b4) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!b4\) {$}}
    i = 12;
  } else {
    i = 13;
  }
}

void operator_or() {
  int i = 0;
  bool b5 = (i > 10);
  if (b5 || false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(b5\) {$}}
    i = 14;
  } else {
    i = 15;
  }
  bool b6 = (i > 10);
  if (b6 || true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(true\) {$}}
    i = 16;
  } else {
    i = 17;
  }
  bool b7 = (i > 10);
  if (false || b7) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(b7\) {$}}
    i = 18;
  } else {
    i = 19;
  }
  bool b8 = (i > 10);
  if (true || b8) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(true\) {$}}
    i = 20;
  } else {
    i = 21;
  }
}

void operator_and() {
  int i = 0;
  bool b9 = (i > 20);
  if (b9 && false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(false\) {$}}
    i = 22;
  } else {
    i = 23;
  }
  bool ba = (i > 20);
  if (ba && true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(ba\) {$}}
    i = 24;
  } else {
    i = 25;
  }
  bool bb = (i > 20);
  if (false && bb) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(false\) {$}}
    i = 26;
  } else {
    i = 27;
  }
  bool bc = (i > 20);
  if (true && bc) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(bc\) {$}}
    i = 28;
  } else {
    i = 29;
  }
}

void ternary_operator() {
  int i = 0;
  bool bd = (i > 20) ? true : false;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{^  bool bd = i > 20;$}}

  bool be = (i > 20) ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{^  bool be = i <= 20;$}}

  bool bf = ((i > 20)) ? false : true;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: {{.*}} in ternary expression result
  // CHECK-FIXES: {{^  bool bf = i <= 20;$}}
}

void operator_not_equal() {
  int i = 0;
  bool bf = (i > 20);
  if (false != bf) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(bf\) {$}}
    i = 30;
  } else {
    i = 31;
  }
  bool bg = (i > 20);
  if (true != bg) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!bg\) {$}}
    i = 32;
  } else {
    i = 33;
  }
  bool bh = (i > 20);
  if (bh != false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(bh\) {$}}
    i = 34;
  } else {
    i = 35;
  }
  bool bi = (i > 20);
  if (bi != true) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(!bi\) {$}}
    i = 36;
  } else {
    i = 37;
  }
}

void nested_booleans() {
  if (false || (true || false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(false \|\| \(true\)\) {$}}
  }
  if (true && (true || false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(true && \(true\)\) {$}}
  }
  if (false || (true && false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(false \|\| \(false\)\) {$}}
  }
  if (true && (true && false)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: {{.*}} to boolean operator
    // CHECK-FIXES: {{^  if \(true && \(false\)\) {$}}
  }
}

static constexpr bool truthy() {
  return true;
}

#define HAS_XYZ_FEATURE true

void macros_and_constexprs(int i = 0) {
  bool b = (i == 1);
  if (b && truthy()) {
    // leave this alone; if you want it simplified, then you should
    // inline the constexpr function first.
    i = 1;
  }
  i = 2;
  if (b && HAS_XYZ_FEATURE) {
    // leave this alone; if you want it simplified, then you should
    // inline the macro first.
    i = 3;
  }
  i = 4;
}

bool conditional_return_statements(int i) {
  if (i == 0) return true; else return false;
}
// CHECK-MESSAGES: :[[@LINE-2]]:22: warning: {{.*}} in conditional return statement
// CHECK-FIXES:      {{^}}  return i == 0;{{$}}
// CHECK-FIXES-NEXT: {{^}$}}

bool conditional_return_statements_then_expr(int i, int j) {
  if (i == j) return (i == 0); else return false;
}

bool conditional_return_statements_else_expr(int i, int j) {
  if (i == j) return true; else return (i == 0);
}

bool negated_conditional_return_statements(int i) {
  if (i == 0) return false; else return true;
}
// CHECK-MESSAGES: :[[@LINE-2]]:22: warning: {{.*}} in conditional return statement
// CHECK-FIXES:      {{^}}  return i != 0;{{$}}
// CHECK-FIXES-NEXT: {{^}$}}

bool conditional_compound_return_statements(int i) {
  if (i == 1) {
    return true;
  } else {
    return false;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return statement
// CHECK-FIXES:      {{^}}bool conditional_compound_return_statements(int i) {{{$}}
// CHECK-FIXES-NEXT: {{^}}  return i == 1;{{$}}
// CHECK-FIXES-NEXT: {{^}$}}

bool negated_conditional_compound_return_statements(int i) {
  if (i == 1) {
    return false;
  } else {
    return true;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:12: warning: {{.*}} in conditional return statement
// CHECK-FIXES:      {{^}}bool negated_conditional_compound_return_statements(int i) {{{$}}
// CHECK-FIXES-NEXT: {{^}}  return i != 1;{{$}}
// CHECK-FIXES-NEXT: {{^}$}}

bool conditional_return_statements_side_effects_then(int i) {
  if (i == 2) {
    macros_and_constexprs();
    return true;
  } else
    return false;
}

bool negated_conditional_return_statements_side_effects_then(int i) {
  if (i == 2) {
    macros_and_constexprs();
    return false;
  } else
    return true;
}

bool conditional_return_statements_side_effects_else(int i) {
  if (i == 2)
    return true;
  else {
    macros_and_constexprs();
    return false;
  }
}

bool negated_conditional_return_statements_side_effects_else(int i) {
  if (i == 2)
    return false;
  else {
    macros_and_constexprs();
    return true;
  }
}

void lambda_conditional_return_statements() {
  auto lambda = [](int n) -> bool { if (n > 0) return true; else return false; };
  // CHECK-MESSAGES: :[[@LINE-1]]:55: warning: {{.*}} in conditional return statement
  // CHECK-FIXES: {{^}}  auto lambda = [](int n) -> bool { return n > 0; };{{$}}

  auto lambda2 = [](int n) -> bool {
    if (n > 0) {
        return true;
    } else {
        return false;
    }
  };
  // CHECK-MESSAGES: :[[@LINE-5]]:16: warning: {{.*}} in conditional return statement
  // CHECK-FIXES:      {{^}}  auto lambda2 = [](int n) -> bool {{{$}}
  // CHECK-FIXES-NEXT: {{^}}    return n > 0;{{$}}
  // CHECK-FIXES-NEXT: {{^}}  };{{$}}

  auto lambda3 = [](int n) -> bool { if (n > 0) {macros_and_constexprs(); return true; } else return false; };

  auto lambda4 = [](int n) -> bool {
    if (n > 0)
        return true;
    else {
        macros_and_constexprs();
        return false;
    }
  };

  auto lambda5 = [](int n) -> bool { if (n > 0) return false; else return true; };
  // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: {{.*}} in conditional return statement
  // CHECK-FIXES: {{^}}  auto lambda5 = [](int n) -> bool { return n <= 0; };{{$}}

  auto lambda6 = [](int n) -> bool {
    if (n > 0) {
        return false;
    } else {
        return true;
    }
  };
  // CHECK-MESSAGES: :[[@LINE-5]]:16: warning: {{.*}} in conditional return statement
  // CHECK-FIXES:      {{^}}  auto lambda6 = [](int n) -> bool {{{$}}
  // CHECK-FIXES-NEXT: {{^}}    return n <= 0;{{$}}
  // CHECK-FIXES-NEXT: {{^}}  };{{$}}
}

void simple_conditional_assignment_statements(int i) {
  bool b;
  if (i > 10)
    b = true;
  else
    b = false;
  bool bb = false;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: bool b;
  // CHECK-FIXES: {{^  }}b = i > 10;{{$}}
  // CHECK-FIXES: bool bb = false;

  bool c;
  if (i > 20)
    c = false;
  else
    c = true;
  bool c2 = false;
  // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: bool c;
  // CHECK-FIXES: {{^  }}c = i <= 20;{{$}}
  // CHECK-FIXES: bool c2 = false;

  // unchanged; different variables
  bool b2;
  if (i > 12)
    b = true;
  else
    b2 = false;

  // unchanged; no else statement
  bool b3;
  if (i > 15)
    b3 = true;

  // unchanged; not boolean assignment
  int j;
  if (i > 17)
    j = 10;
  else
    j = 20;

  // unchanged; different variables assigned
  int k = 0;
  bool b4 = false;
  if (i > 10)
    b4 = true;
  else
    k = 10;
}

void complex_conditional_assignment_statements(int i) {
  bool d;
  if (i > 30) {
    d = true;
  } else {
    d = false;
  }
  d = false;
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: bool d;
  // CHECK-FIXES: {{^  }}d = i > 30;{{$}}
  // CHECK-FIXES: d = false;

  bool e;
  if (i > 40) {
    e = false;
  } else {
    e = true;
  }
  e = false;
  // CHECK-MESSAGES: :[[@LINE-5]]:9: warning: {{.*}} in conditional assignment
  // CHECK-FIXES: bool e;
  // CHECK-FIXES: {{^  }}e = i <= 40;{{$}}
  // CHECK-FIXES: e = false;

  // unchanged; no else statement
  bool b3;
  if (i > 15) {
    b3 = true;
  }

  // unchanged; not a boolean assignment
  int j;
  if (i > 17) {
    j = 10;
  } else {
    j = 20;
  }

  // unchanged; multiple statements
  bool f;
  if (j > 10) {
    j = 10;
    f = true;
  } else {
    j = 20;
    f = false;
  }

  // unchanged; multiple statements
  bool g;
  if (j > 10)
    f = true;
  else {
    j = 20;
    f = false;
  }

  // unchanged; multiple statements
  bool h;
  if (j > 10) {
    j = 10;
    f = true;
  } else
    f = false;
}
