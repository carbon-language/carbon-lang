// RUN: $(dirname %s)/check_clang_tidy.sh %s readability-simplify-boolean-expr %t -config="{CheckOptions: [{key: "readability-simplify-boolean-expr.ChainedConditionalReturn", value: 1}]}" --
// REQUIRES: shell

bool chained_conditional_compound_return(int i) {
  if (i < 0) {
    return true;
  } else if (i < 10) {
    return false;
  } else if (i > 20) {
    return true;
  } else {
    return false;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:12: warning: redundant boolean literal in conditional return statement [readability-simplify-boolean-expr]
  // CHECK-FIXES:      {{^}}  } else if (i < 10) {{{$}}
  // CHECK-FIXES-NEXT: {{^}}    return false;{{$}}
  // CHECK-FIXES-NEXT: {{^}}  } else return i > 20;{{$}}
}

bool chained_conditional_return(int i) {
  if (i < 0)
    return true;
  else if (i < 10)
    return false;
  else if (i > 20)
    return true;
  else
    return false;
  // CHECK-MESSAGES: :[[@LINE-3]]:12: warning: {{.*}} in conditional return statement
  // CHECK-FIXES:      {{^}}  else if (i < 10)
  // CHECK-FIXES-NEXT: {{^}}    return false;
  // CHECK-FIXES-NEXT: {{^}}  else return i > 20;
}
