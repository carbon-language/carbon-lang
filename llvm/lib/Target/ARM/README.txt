//===---------------------------------------------------------------------===//
// Random ideas for the ARM backend.
//===---------------------------------------------------------------------===//

Consider implementing a select with two conditional moves:

cmp x, y
moveq dst, a
movne dst, b
