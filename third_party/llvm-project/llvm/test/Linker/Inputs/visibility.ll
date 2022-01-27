$c1 = comdat any

; Variables
@v1 = weak hidden global i32 0
@v2 = weak protected global i32 0
@v3 = weak hidden global i32 0
@v4 = hidden global i32 1, comdat($c1)

; Aliases
@a1 = weak hidden alias i32, i32* @v1
@a2 = weak protected alias i32, i32* @v2
@a3 = weak hidden alias i32, i32* @v3

; Functions
define weak hidden void @f1() {
entry:
  ret void
}
define weak protected void @f2() {
entry:
  ret void
}
define weak hidden void @f3() {
entry:
  ret void
}
