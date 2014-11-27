// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"runtime"
	"unsafe"
)

// The ffi function, written in C, allocates an FFI closure.  It
// returns the code and data pointers.  When the code pointer is
// called, it will call callback.  CIF is an FFI data structure
// allocated as part of the closure, and is returned to ensure that
// the GC retains it.
func ffi(ftyp *funcType, callback func(unsafe.Pointer, unsafe.Pointer)) (code uintptr, data uintptr, cif unsafe.Pointer)

// The ffiFree function, written in C, releases the FFI closure.
func ffiFree(uintptr)

// An ffiData holds the information needed to preserve an FFI closure
// for the garbage collector.
type ffiData struct {
	code     uintptr
	data     uintptr
	cif      unsafe.Pointer
	callback func(unsafe.Pointer, unsafe.Pointer)
}

// The makeFuncFFI function uses libffi closures to implement
// reflect.MakeFunc.  This is used for processors for which we don't
// have more efficient support.
func makeFuncFFI(ftyp *funcType, fn func(args []Value) (results []Value)) (uintptr, *ffiData) {
	callback := func(params, results unsafe.Pointer) {
		ffiCall(ftyp, fn, params, results)
	}

	code, data, cif := ffi(ftyp, callback)

	c := &ffiData{code: code, data: data, cif: cif, callback: callback}

	runtime.SetFinalizer(c,
		func(p *ffiData) {
			ffiFree(p.data)
		})

	return code, c
}

// ffiCall takes pointers to the parameters, calls the function, and
// stores the results back into memory.
func ffiCall(ftyp *funcType, fn func([]Value) []Value, params unsafe.Pointer, results unsafe.Pointer) {
	in := make([]Value, 0, len(ftyp.in))
	ap := params
	for _, rt := range ftyp.in {
		p := unsafe_New(rt)
		memmove(p, *(*unsafe.Pointer)(ap), rt.size)
		v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
		in = append(in, v)
		ap = (unsafe.Pointer)(uintptr(ap) + ptrSize)
	}

	out := fn(in)

	off := uintptr(0)
	for i, typ := range ftyp.out {
		v := out[i]
		if v.typ != typ {
			panic("reflect: function created by MakeFunc using " + funcName(fn) +
				" returned wrong type: have " +
				out[i].typ.String() + " for " + typ.String())
		}
		if v.flag&flagRO != 0 {
			panic("reflect: function created by MakeFunc using " + funcName(fn) +
				" returned value obtained from unexported field")
		}

		off = align(off, uintptr(typ.fieldAlign))
		addr := unsafe.Pointer(uintptr(results) + off)
		if v.flag&flagIndir == 0 && (v.kind() == Ptr || v.kind() == UnsafePointer) {
			*(*unsafe.Pointer)(addr) = v.ptr
		} else {
			memmove(addr, v.ptr, typ.size)
		}
		off += typ.size
	}
}
