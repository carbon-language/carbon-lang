// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

// Unix environment variables.

package syscall

import "sync"

var (
	// envOnce guards initialization by copyenv, which populates env.
	envOnce sync.Once

	// envLock guards env and envs.
	envLock sync.RWMutex

	// env maps from an environment variable to its first occurrence in envs.
	env map[string]int

	// envs is provided by the runtime. elements are expected to be
	// of the form "key=value".
	Envs []string
)

// setenv_c is provided by the runtime, but is a no-op if cgo isn't
// loaded.
func setenv_c(k, v string)

func copyenv() {
	env = make(map[string]int)
	for i, s := range Envs {
		for j := 0; j < len(s); j++ {
			if s[j] == '=' {
				key := s[:j]
				if _, ok := env[key]; !ok {
					env[key] = i
				}
				break
			}
		}
	}
}

func Getenv(key string) (value string, found bool) {
	envOnce.Do(copyenv)
	if len(key) == 0 {
		return "", false
	}

	envLock.RLock()
	defer envLock.RUnlock()

	i, ok := env[key]
	if !ok {
		return "", false
	}
	s := Envs[i]
	for i := 0; i < len(s); i++ {
		if s[i] == '=' {
			return s[i+1:], true
		}
	}
	return "", false
}

func Setenv(key, value string) error {
	envOnce.Do(copyenv)
	if len(key) == 0 {
		return EINVAL
	}
	for i := 0; i < len(key); i++ {
		if key[i] == '=' || key[i] == 0 {
			return EINVAL
		}
	}
	for i := 0; i < len(value); i++ {
		if value[i] == 0 {
			return EINVAL
		}
	}

	envLock.Lock()
	defer envLock.Unlock()

	i, ok := env[key]
	kv := key + "=" + value
	if ok {
		Envs[i] = kv
	} else {
		i = len(Envs)
		Envs = append(Envs, kv)
	}
	env[key] = i
	setenv_c(key, value)
	return nil
}

func Clearenv() {
	envOnce.Do(copyenv) // prevent copyenv in Getenv/Setenv

	envLock.Lock()
	defer envLock.Unlock()

	env = make(map[string]int)
	Envs = []string{}
	// TODO(bradfitz): pass through to C
}

func Environ() []string {
	envOnce.Do(copyenv)
	envLock.RLock()
	defer envLock.RUnlock()
	a := make([]string, len(Envs))
	copy(a, Envs)
	return a
}
