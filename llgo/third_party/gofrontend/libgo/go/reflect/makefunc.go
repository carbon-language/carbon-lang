// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MakeFunc implementation.

package reflect

import (
	"runtime"
	"unsafe"
)

// makeFuncImpl is the closure value implementing the function
// returned by MakeFunc.
type makeFuncImpl struct {
	code uintptr
	typ  *funcType
	fn   func([]Value) []Value

	// For gccgo we use the same entry point for functions and for
	// method values.
	method int
	rcvr   Value

	// When using FFI, hold onto the FFI closure for the garbage
	// collector.
	ffi *ffiData
}

// MakeFunc returns a new function of the given Type
// that wraps the function fn. When called, that new function
// does the following:
//
//	- converts its arguments to a slice of Values.
//	- runs results := fn(args).
//	- returns the results as a slice of Values, one per formal result.
//
// The implementation fn can assume that the argument Value slice
// has the number and type of arguments given by typ.
// If typ describes a variadic function, the final Value is itself
// a slice representing the variadic arguments, as in the
// body of a variadic function. The result Value slice returned by fn
// must have the number and type of results given by typ.
//
// The Value.Call method allows the caller to invoke a typed function
// in terms of Values; in contrast, MakeFunc allows the caller to implement
// a typed function in terms of Values.
//
// The Examples section of the documentation includes an illustration
// of how to use MakeFunc to build a swap function for different types.
//
func MakeFunc(typ Type, fn func(args []Value) (results []Value)) Value {
	if typ.Kind() != Func {
		panic("reflect: call of MakeFunc with non-Func type")
	}

	t := typ.common()
	ftyp := (*funcType)(unsafe.Pointer(t))

	var code uintptr
	var ffi *ffiData
	switch runtime.GOARCH {
	case "amd64", "386", "s390", "s390x":
		// Indirect Go func value (dummy) to obtain actual
		// code address. (A Go func value is a pointer to a C
		// function pointer. http://golang.org/s/go11func.)
		dummy := makeFuncStub
		code = **(**uintptr)(unsafe.Pointer(&dummy))
	default:
		code, ffi = makeFuncFFI(ftyp, fn)
	}

	impl := &makeFuncImpl{
		code:   code,
		typ:    ftyp,
		fn:     fn,
		method: -1,
		ffi:    ffi,
	}

	return Value{t, unsafe.Pointer(&impl), flag(Func<<flagKindShift) | flagIndir}
}

// makeFuncStub is an assembly function that is the code half of
// the function returned from MakeFunc. It expects a *callReflectFunc
// as its context register, and its job is to invoke callReflect(ctxt, frame)
// where ctxt is the context register and frame is a pointer to the first
// word in the passed-in argument frame.
func makeFuncStub()

// makeMethodValue converts v from the rcvr+method index representation
// of a method value to an actual method func value, which is
// basically the receiver value with a special bit set, into a true
// func value - a value holding an actual func. The output is
// semantically equivalent to the input as far as the user of package
// reflect can tell, but the true func representation can be handled
// by code like Convert and Interface and Assign.
func makeMethodValue(op string, v Value) Value {
	if v.flag&flagMethod == 0 {
		panic("reflect: internal error: invalid use of makeMethodValue")
	}

	// Ignoring the flagMethod bit, v describes the receiver, not the method type.
	fl := v.flag & (flagRO | flagAddr | flagIndir)
	fl |= flag(v.typ.Kind()) << flagKindShift
	rcvr := Value{v.typ, v.ptr /* v.scalar, */, fl}

	// v.Type returns the actual type of the method value.
	ft := v.Type().(*rtype)

	// Cause panic if method is not appropriate.
	// The panic would still happen during the call if we omit this,
	// but we want Interface() and other operations to fail early.
	_, t, _ := methodReceiver(op, rcvr, int(v.flag)>>flagMethodShift)

	ftyp := (*funcType)(unsafe.Pointer(t))
	method := int(v.flag) >> flagMethodShift

	fv := &makeFuncImpl{
		typ:    ftyp,
		method: method,
		rcvr:   rcvr,
	}

	switch runtime.GOARCH {
	case "amd64", "386":
		// Indirect Go func value (dummy) to obtain actual
		// code address. (A Go func value is a pointer to a C
		// function pointer. http://golang.org/s/go11func.)
		dummy := makeFuncStub
		fv.code = **(**uintptr)(unsafe.Pointer(&dummy))
	default:
		fv.code, fv.ffi = makeFuncFFI(ftyp, fv.call)
	}

	return Value{ft, unsafe.Pointer(&fv), v.flag&flagRO | flag(Func)<<flagKindShift | flagIndir}
}

// makeValueMethod takes a method function and returns a function that
// takes a value receiver and calls the real method with a pointer to
// it.
func makeValueMethod(v Value) Value {
	typ := v.typ
	if typ.Kind() != Func {
		panic("reflect: call of makeValueMethod with non-Func type")
	}
	if v.flag&flagMethodFn == 0 {
		panic("reflect: call of makeValueMethod with non-MethodFn")
	}

	t := typ.common()
	ftyp := (*funcType)(unsafe.Pointer(t))

	impl := &makeFuncImpl{
		typ:    ftyp,
		method: -2,
		rcvr:   v,
	}

	switch runtime.GOARCH {
	case "amd64", "386", "s390", "s390x":
		// Indirect Go func value (dummy) to obtain actual
		// code address. (A Go func value is a pointer to a C
		// function pointer. http://golang.org/s/go11func.)
		dummy := makeFuncStub
		impl.code = **(**uintptr)(unsafe.Pointer(&dummy))
	default:
		impl.code, impl.ffi = makeFuncFFI(ftyp, impl.call)
	}

	return Value{t, unsafe.Pointer(&impl), flag(Func<<flagKindShift) | flagIndir}
}

// Call the function represented by a makeFuncImpl.
func (c *makeFuncImpl) call(in []Value) []Value {
	if c.method == -1 {
		return c.fn(in)
	} else if c.method == -2 {
		if c.typ.IsVariadic() {
			return c.rcvr.CallSlice(in)
		} else {
			return c.rcvr.Call(in)
		}
	} else {
		m := c.rcvr.Method(c.method)
		if c.typ.IsVariadic() {
			return m.CallSlice(in)
		} else {
			return m.Call(in)
		}
	}
}
