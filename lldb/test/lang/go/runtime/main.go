package main

import "fmt"

type Fooer interface {
	Foo() int
}

type SomeFooer struct {
	val int
}

func (s SomeFooer) Foo() int {
	return s.val
}

type AnotherFooer struct {
    a, b, c int
}

func (s AnotherFooer) Foo() int {
	return s.a
}


func printEface(a, b, c, d interface{}) {
    fmt.Println(a, b, c, d)  // Set breakpoint 1
}

func printIface(a, b Fooer) {
    fmt.Println(a, b)  // Set breakpoint 2
}
func main() {
    sf := SomeFooer{9}
    af := AnotherFooer{-1, -2, -3}
    printEface(1,2.0, sf, af)
    printIface(sf, af)
}