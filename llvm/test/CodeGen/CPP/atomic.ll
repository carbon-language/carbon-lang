; RUN: llc -march=cpp -o - %s | FileCheck %s

define void @test_atomicrmw(i32* %addr, i32 %inc) {
  %inst0 = atomicrmw xchg i32* %addr, i32 %inc seq_cst
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Xchg, {{.*}}, SequentiallyConsistent, CrossThread
  ; CHECK: [[INST]]->setName("inst0");
  ; CHECK: [[INST]]->setVolatile(false);

  %inst1 = atomicrmw add i32* %addr, i32 %inc seq_cst
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Add, {{.*}}, SequentiallyConsistent, CrossThread
  ; CHECK: [[INST]]->setName("inst1");
  ; CHECK: [[INST]]->setVolatile(false);

  %inst2 = atomicrmw volatile sub i32* %addr, i32 %inc singlethread monotonic
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Sub, {{.*}}, Monotonic, SingleThread
  ; CHECK: [[INST]]->setName("inst2");
  ; CHECK: [[INST]]->setVolatile(true);

  %inst3 = atomicrmw and i32* %addr, i32 %inc acq_rel
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::And, {{.*}}, AcquireRelease, CrossThread
  ; CHECK: [[INST]]->setName("inst3");
  ; CHECK: [[INST]]->setVolatile(false);

  %inst4 = atomicrmw nand i32* %addr, i32 %inc release
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Nand, {{.*}}, Release, CrossThread
  ; CHECK: [[INST]]->setName("inst4");
  ; CHECK: [[INST]]->setVolatile(false);

  %inst5 = atomicrmw volatile or i32* %addr, i32 %inc singlethread seq_cst
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Or, {{.*}}, SequentiallyConsistent, SingleThread
  ; CHECK: [[INST]]->setName("inst5");
  ; CHECK: [[INST]]->setVolatile(true);

  %inst6 = atomicrmw xor i32* %addr, i32 %inc release
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Xor, {{.*}}, Release, CrossThread
  ; CHECK: [[INST]]->setName("inst6");
  ; CHECK: [[INST]]->setVolatile(false);

  %inst7 = atomicrmw volatile max i32* %addr, i32 %inc singlethread monotonic
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Max, {{.*}}, Monotonic, SingleThread
  ; CHECK: [[INST]]->setName("inst7");
  ; CHECK: [[INST]]->setVolatile(true);

  %inst8 = atomicrmw min i32* %addr, i32 %inc acquire
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::Min, {{.*}}, Acquire, CrossThread
  ; CHECK: [[INST]]->setName("inst8");
  ; CHECK: [[INST]]->setVolatile(false);

  %inst9 = atomicrmw volatile umax i32* %addr, i32 %inc monotonic
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::UMax, {{.*}}, Monotonic, CrossThread
  ; CHECK: [[INST]]->setName("inst9");
  ; CHECK: [[INST]]->setVolatile(true);

  %inst10 = atomicrmw umin i32* %addr, i32 %inc singlethread release
  ; CHECK: AtomicRMWInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicRMWInst(AtomicRMWInst::UMin, {{.*}}, Release, SingleThread
  ; CHECK: [[INST]]->setName("inst10");
  ; CHECK: [[INST]]->setVolatile(false);


  ret void
}

define void @test_cmpxchg(i32* %addr, i32 %desired, i32 %new) {
  %inst0 = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  ; CHECK: AtomicCmpXchgInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicCmpXchgInst({{.*}}, SequentiallyConsistent, Monotonic, CrossThread
  ; CHECK: [[INST]]->setName("inst0");
  ; CHECK: [[INST]]->setVolatile(false);
  ; CHECK: [[INST]]->setWeak(false);

  %inst1 = cmpxchg volatile i32* %addr, i32 %desired, i32 %new singlethread acq_rel acquire
  ; CHECK: AtomicCmpXchgInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicCmpXchgInst({{.*}}, AcquireRelease, Acquire, SingleThread
  ; CHECK: [[INST]]->setName("inst1");
  ; CHECK: [[INST]]->setVolatile(true);
  ; CHECK: [[INST]]->setWeak(false);

  %inst2 = cmpxchg weak i32* %addr, i32 %desired, i32 %new seq_cst monotonic
  ; CHECK: AtomicCmpXchgInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicCmpXchgInst({{.*}}, SequentiallyConsistent, Monotonic, CrossThread
  ; CHECK: [[INST]]->setName("inst2");
  ; CHECK: [[INST]]->setVolatile(false);
  ; CHECK: [[INST]]->setWeak(true);

  %inst3 = cmpxchg weak volatile i32* %addr, i32 %desired, i32 %new singlethread acq_rel acquire
  ; CHECK: AtomicCmpXchgInst* [[INST:[a-zA-Z0-9_]+]] = new AtomicCmpXchgInst({{.*}}, AcquireRelease, Acquire, SingleThread
  ; CHECK: [[INST]]->setName("inst3");
  ; CHECK: [[INST]]->setVolatile(true);
  ; CHECK: [[INST]]->setWeak(true);

  ret void
}
