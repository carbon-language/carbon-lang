%t = type { i8* }
declare %t @f()

define %t @g() {
 %x = call %t @f()
 ret %t %x
}
