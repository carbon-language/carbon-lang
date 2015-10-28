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

type mystruct struct {
	myInt int
	myPointer *mystruct
}

func main() {
	theBool := true
	theInt := -7
	theComplex := 1 + 2i
	pointee := -10
	thePointer := &pointee
	theStruct := &mystruct { myInt: 7}
	theStruct.myPointer = theStruct
	theArray := [5]byte{1, 2, 3, 4, 5}
	theSlice := theArray[1:2]
	theString := "abc"
	
	f := SomeFooer {9}
	var theEface interface{} = f
	var theFooer Fooer = f
	
	theChan := make(chan int)
	theMap := make(map[int]string)
	theMap[1] = "1"

	fmt.Println(theBool)  // Set breakpoint here.
	// Reference all the variables so the compiler is happy.
	fmt.Println(theInt, theComplex, thePointer, theStruct.myInt)
	fmt.Println(theArray[0], theSlice[0], theString)
	fmt.Println(theEface, theFooer, theChan, theMap)
}