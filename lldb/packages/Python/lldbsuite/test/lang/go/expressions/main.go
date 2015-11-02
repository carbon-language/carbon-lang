package main

import "fmt"

type myStruct struct {
    a, b int
}

var myGlobal = 17

func myFunc(i interface{}) {
    a := [...]int{8, 9, 10}
    b := a[:]
    x := 22
    fmt.Println(a, b, x, i, myGlobal)  // Set breakpoint here.
}

func main() {
    s := myStruct {2, -1}
    myFunc(s)
}