// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !netgo
// +build darwin dragonfly freebsd linux netbsd openbsd

package net

/*
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
*/

import (
	"syscall"
	"unsafe"
)

//extern getaddrinfo
func libc_getaddrinfo(node *byte, service *byte, hints *syscall.Addrinfo, res **syscall.Addrinfo) int

//extern freeaddrinfo
func libc_freeaddrinfo(res *syscall.Addrinfo)

//extern gai_strerror
func libc_gai_strerror(errcode int) *byte

// bytePtrToString takes a NUL-terminated array of bytes and convert
// it to a Go string.
func bytePtrToString(p *byte) string {
	a := (*[10000]byte)(unsafe.Pointer(p))
	i := 0
	for a[i] != 0 {
		i++
	}
	return string(a[:i])
}

func cgoLookupHost(name string) (addrs []string, err error, completed bool) {
	ip, err, completed := cgoLookupIP(name)
	for _, p := range ip {
		addrs = append(addrs, p.String())
	}
	return
}

func cgoLookupPort(net, service string) (port int, err error, completed bool) {
	acquireThread()
	defer releaseThread()

	var res *syscall.Addrinfo
	var hints syscall.Addrinfo

	switch net {
	case "":
		// no hints
	case "tcp", "tcp4", "tcp6":
		hints.Ai_socktype = syscall.SOCK_STREAM
		hints.Ai_protocol = syscall.IPPROTO_TCP
	case "udp", "udp4", "udp6":
		hints.Ai_socktype = syscall.SOCK_DGRAM
		hints.Ai_protocol = syscall.IPPROTO_UDP
	default:
		return 0, UnknownNetworkError(net), true
	}
	if len(net) >= 4 {
		switch net[3] {
		case '4':
			hints.Ai_family = syscall.AF_INET
		case '6':
			hints.Ai_family = syscall.AF_INET6
		}
	}

	s := syscall.StringBytePtr(service)
	syscall.Entersyscall()
	gerrno := libc_getaddrinfo(nil, s, &hints, &res)
	syscall.Exitsyscall()
	if gerrno == 0 {
		defer libc_freeaddrinfo(res)
		for r := res; r != nil; r = r.Ai_next {
			switch r.Ai_family {
			default:
				continue
			case syscall.AF_INET:
				sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.Ai_addr))
				p := (*[2]byte)(unsafe.Pointer(&sa.Port))
				return int(p[0])<<8 | int(p[1]), nil, true
			case syscall.AF_INET6:
				sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.Ai_addr))
				p := (*[2]byte)(unsafe.Pointer(&sa.Port))
				return int(p[0])<<8 | int(p[1]), nil, true
			}
		}
	}
	return 0, &AddrError{"unknown port", net + "/" + service}, true
}

func cgoLookupIPCNAME(name string) (addrs []IP, cname string, err error, completed bool) {
	acquireThread()
	defer releaseThread()

	var res *syscall.Addrinfo
	var hints syscall.Addrinfo

	hints.Ai_flags = int32(cgoAddrInfoFlags())
	hints.Ai_socktype = syscall.SOCK_STREAM

	h := syscall.StringBytePtr(name)
	syscall.Entersyscall()
	gerrno := libc_getaddrinfo(h, nil, &hints, &res)
	syscall.Exitsyscall()
	if gerrno != 0 {
		var str string
		if gerrno == syscall.EAI_NONAME {
			str = noSuchHost
		} else if gerrno == syscall.EAI_SYSTEM {
			errno := syscall.GetErrno()
			if errno == 0 {
				// err should not be nil, but sometimes getaddrinfo returns
				// gerrno == C.EAI_SYSTEM with err == nil on Linux.
				// The report claims that it happens when we have too many
				// open files, so use syscall.EMFILE (too many open files in system).
				// Most system calls would return ENFILE (too many open files),
				// so at the least EMFILE should be easy to recognize if this
				// comes up again. golang.org/issue/6232.
				errno = syscall.EMFILE
			}
			str = errno.Error()
		} else {
			str = bytePtrToString(libc_gai_strerror(gerrno))
		}
		return nil, "", &DNSError{Err: str, Name: name}, true
	}
	defer libc_freeaddrinfo(res)
	if res != nil {
		cname = bytePtrToString((*byte)(unsafe.Pointer(res.Ai_canonname)))
		if cname == "" {
			cname = name
		}
		if len(cname) > 0 && cname[len(cname)-1] != '.' {
			cname += "."
		}
	}
	for r := res; r != nil; r = r.Ai_next {
		// We only asked for SOCK_STREAM, but check anyhow.
		if r.Ai_socktype != syscall.SOCK_STREAM {
			continue
		}
		switch r.Ai_family {
		default:
			continue
		case syscall.AF_INET:
			sa := (*syscall.RawSockaddrInet4)(unsafe.Pointer(r.Ai_addr))
			addrs = append(addrs, copyIP(sa.Addr[:]))
		case syscall.AF_INET6:
			sa := (*syscall.RawSockaddrInet6)(unsafe.Pointer(r.Ai_addr))
			addrs = append(addrs, copyIP(sa.Addr[:]))
		}
	}
	return addrs, cname, nil, true
}

func cgoLookupIP(name string) (addrs []IP, err error, completed bool) {
	addrs, _, err, completed = cgoLookupIPCNAME(name)
	return
}

func cgoLookupCNAME(name string) (cname string, err error, completed bool) {
	_, cname, err, completed = cgoLookupIPCNAME(name)
	return
}

func copyIP(x IP) IP {
	if len(x) < 16 {
		return x.To16()
	}
	y := make(IP, len(x))
	copy(y, x)
	return y
}
