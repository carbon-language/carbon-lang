// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Definitions related to data race detection.

#ifdef RACE
enum { raceenabled = 1 };
#else
enum { raceenabled = 0 };
#endif

// Initialize race detection subsystem.
uintptr	runtime_raceinit(void);
// Finalize race detection subsystem, does not return.
void	runtime_racefini(void);

void	runtime_racemapshadow(void *addr, uintptr size);
void	runtime_racemalloc(void *p, uintptr sz);
uintptr	runtime_racegostart(void *pc);
void	runtime_racegoend(void);
void	runtime_racewritepc(void *addr, void *callpc, void *pc);
void	runtime_racereadpc(void *addr, void *callpc, void *pc);
void	runtime_racewriterangepc(void *addr, uintptr sz, void *callpc, void *pc);
void	runtime_racereadrangepc(void *addr, uintptr sz, void *callpc, void *pc);
void	runtime_racereadobjectpc(void *addr, const Type *t, void *callpc, void *pc);
void	runtime_racewriteobjectpc(void *addr, const Type *t, void *callpc, void *pc);
void	runtime_racefingo(void);
void	runtime_raceacquire(void *addr);
void	runtime_raceacquireg(G *gp, void *addr);
void	runtime_racerelease(void *addr);
void	runtime_racereleaseg(G *gp, void *addr);
void	runtime_racereleasemerge(void *addr);
void	runtime_racereleasemergeg(G *gp, void *addr);
