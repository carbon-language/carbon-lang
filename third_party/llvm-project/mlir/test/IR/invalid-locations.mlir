// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @location_missing_l_paren() {
^bb:
  return loc) // expected-error {{expected '(' in location}}
}

// -----

func @location_missing_r_paren() {
^bb:
  return loc(unknown // expected-error@+1 {{expected ')' in location}}
}

// -----

func @location_invalid_instance() {
^bb:
  return loc() // expected-error {{expected location instance}}
}

// -----

func @location_name_missing_r_paren() {
^bb:
  return loc("foo"(unknown]) // expected-error {{expected ')' after child location of NameLoc}}
}

// -----

func @location_name_child_is_name() {
^bb:
  return loc("foo"("foo")) // expected-error {{child of NameLoc cannot be another NameLoc}}
}

// -----

func @location_callsite_missing_l_paren() {
^bb:
  return loc(callsite unknown  // expected-error {{expected '(' in callsite location}}
}

// -----

func @location_callsite_missing_callee() {
^bb:
  return loc(callsite( at )  // expected-error {{expected location instance}}
}

// -----

func @location_callsite_missing_at() {
^bb:
  return loc(callsite(unknown unknown) // expected-error {{expected 'at' in callsite location}}
}

// -----

func @location_callsite_missing_caller() {
^bb:
  return loc(callsite(unknown at )  // expected-error {{expected location instance}}
}

// -----

func @location_callsite_missing_r_paren() {
^bb:
  return loc(callsite( unknown at unknown  // expected-error@+1 {{expected ')' in callsite location}}
}

// -----

func @location_fused_missing_greater() {
^bb:
  return loc(fused<true [unknown]) // expected-error {{expected '>' after fused location metadata}}
}

// -----

func @location_fused_missing_metadata() {
^bb:
  // expected-error@+1 {{expected non-function type}}
  return loc(fused<) // expected-error {{expected valid attribute metadata}}
}

// -----

func @location_fused_missing_l_square() {
^bb:
  return loc(fused<true>unknown]) // expected-error {{expected '[' in fused location}}
}

// -----

func @location_fused_missing_r_square() {
^bb:
  return loc(fused[unknown) // expected-error {{expected ']' in fused location}}
}

// -----

func @location_invalid_alias() {
  // expected-error@+1 {{expected location, but found dialect attribute: '#foo.loc'}}
  return loc(#foo.loc)
}

// -----

func @location_invalid_alias() {
  // expected-error@+1 {{operation location alias was never defined}}
  return loc(#invalid_alias)
}

// -----

func @location_invalid_alias() {
  // expected-error@+1 {{expected location, but found 'true'}}
  return loc(#non_loc)
}

#non_loc = true

