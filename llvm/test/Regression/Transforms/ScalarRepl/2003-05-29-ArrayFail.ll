; RUN: as < %s | opt -scalarrepl | dis | grep alloca | grep '4 x'

; Test that an array is not incorrectly deconstructed...

int %test() {
  %X = alloca [4 x int]
  %Y = getelementptr [4 x int]* %X, long 0, long 0
  %Z = getelementptr int* %Y, long 1           ; Must preserve arrayness!
  %A = load int* %Z
  ret int %A
}
