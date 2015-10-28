package main

import (
	"fmt"
	"runtime"
)

type philosopher struct {
	i int
	forks [2]chan bool
	eating chan int
	done  chan struct{}
}

func (p philosopher) run() {
	for {
		select {
		case <-p.done:
			return
		case <-p.forks[0]:
			p.eat()
		}
	}
}

func (p philosopher) eat() {
	select {
	case <-p.done:
		return
	case <-p.forks[1]:
		p.eating <- p.i
		p.forks[0] <- true
		p.forks[1] <- true
		runtime.Gosched()
	}
}

func startPhilosophers(n int) (chan struct{}, chan int) {
	philosophers := make([]*philosopher, n)
	chans := make([]chan bool, n)
	for i := range chans {
		chans[i] = make(chan bool, 1)
		chans[i] <- true
	}
	eating := make(chan int, n)
	done := make(chan struct{})
	for i := range philosophers {
		var min, max int
		if i == n - 1 {
			min = 0
			max = i
		} else {
			min = i
			max = i + 1
		}
		philosophers[i] = &philosopher{i: i, forks: [2]chan bool{chans[min], chans[max]}, eating: eating, done: done}
		go philosophers[i].run()
	}
	return done, eating
}

func wait(c chan int) {
	fmt.Println(<- c)
	runtime.Gosched()
}

func main() {
	// Restrict go to 1 real thread so we can be sure we're seeing goroutines
	// and not threads.
	runtime.GOMAXPROCS(1)
	// Create a bunch of goroutines
	done, eating := startPhilosophers(20) // stop1
	// Now turn up the number of threads so this goroutine is likely to get
	// scheduled on a different thread.
	runtime.GOMAXPROCS(runtime.NumCPU()) // stop2
	// Now let things run. Hopefully we'll bounce around
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	wait(eating)
	close(done)
	fmt.Println("done") // stop3
}
