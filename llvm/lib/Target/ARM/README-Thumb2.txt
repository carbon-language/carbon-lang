//===---------------------------------------------------------------------===//
// Random ideas for the ARM backend (Thumb2 specific).
//===---------------------------------------------------------------------===//

* We should model IT instructions explicitly. We should introduce them (even if
  if-converter is not run, the function could still contain movcc's) before
  PEI since passes starting from PEI may require exact code size.
