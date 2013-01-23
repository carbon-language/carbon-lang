; RUN: llc -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-X64 %s
; RUN: llc -code-model=kernel -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck --check-prefix=LINUX-KERNEL-X64 %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -o - | FileCheck --check-prefix=DARWIN-X64 %s
; FIXME: Update and expand test when strong heuristic is implemented.

%struct.foo = type { [16 x i8] }
%struct.foo.0 = type { [4 x i8] }

@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; test1a: array of [16 x i8] 
;         no ssp attribute
; Requires no protector.
define void @test1a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test1a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test1a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test1a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test1a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test1b: array of [16 x i8] 
;         ssp attribute
; Requires protector.
define void @test1b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test1b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test1b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test1b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test1b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test1c: array of [16 x i8] 
;         sspstrong attribute
; Requires protector.
define void @test1c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test1c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test1c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test1c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test1c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test1d: array of [16 x i8] 
;         sspreq attribute
; Requires protector.
define void @test1d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test1d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test1d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test1d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test1d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [16 x i8], align 16
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test2a: struct { [16 x i8] }
;         no ssp attribute
; Requires no protector.
define void @test2a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test2a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test2a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test2a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test2a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test2b: struct { [16 x i8] }
;          ssp attribute
; Requires protector.
define void @test2b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test2b:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test2b:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test2b:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test2b:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test2c: struct { [16 x i8] }
;          sspstrong attribute
; Requires protector.
define void @test2c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test2c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test2c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test2c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test2c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test2d: struct { [16 x i8] }
;          sspreq attribute
; Requires protector.
define void @test2d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test2d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test2d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test2d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test2d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [16 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [16 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test3a:  array of [4 x i8]
;          no ssp attribute
; Requires no protector.
define void @test3a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test3a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test3a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test3a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test3a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test3b:  array [4 x i8]
;          ssp attribute
; Requires no protector.
define void @test3b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test3b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test3b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test3b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test3b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test3c:  array of [4 x i8]
;          sspstrong attribute
; Requires protector.
define void @test3c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test3c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test3c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test3c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test3c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test3d:  array of [4 x i8]
;          sspreq attribute
; Requires protector.
define void @test3d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test3d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test3d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test3d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test3d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %buf = alloca [4 x i8], align 1
  store i8* %a, i8** %a.addr, align 8
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %arraydecay1 = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay1)
  ret void
}

; test4a:  struct { [4 x i8] }
;          no ssp attribute
; Requires no protector.
define void @test4a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test4a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test4a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test4a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test4a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test4b:  struct { [4 x i8] }
;          ssp attribute
; Requires no protector.
define void @test4b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test4b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test4b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test4b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test4b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test4c:  struct { [4 x i8] }
;          sspstrong attribute
; Requires protector.
define void @test4c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test4c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test4c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test4c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test4c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test4d:  struct { [4 x i8] }
;          sspreq attribute
; Requires protector.
define void @test4d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test4d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test4d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test4d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test4d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  %b = alloca %struct.foo.0, align 1
  store i8* %a, i8** %a.addr, align 8
  %buf = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay = getelementptr inbounds [4 x i8]* %buf, i32 0, i32 0
  %0 = load i8** %a.addr, align 8
  %call = call i8* @strcpy(i8* %arraydecay, i8* %0)
  %buf1 = getelementptr inbounds %struct.foo.0* %b, i32 0, i32 0
  %arraydecay2 = getelementptr inbounds [4 x i8]* %buf1, i32 0, i32 0
  %call3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %arraydecay2)
  ret void
}

; test5a:  no arrays / no nested arrays
;          no ssp attribute
; Requires no protector.
define void @test5a(i8* %a) nounwind uwtable {
entry:
; LINUX-I386: test5a:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test5a:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test5a:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test5a:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test5b:  no arrays / no nested arrays
;          ssp attribute
; Requires no protector.
define void @test5b(i8* %a) nounwind uwtable ssp {
entry:
; LINUX-I386: test5b:
; LINUX-I386-NOT: calll __stack_chk_fail
; LINUX-I386: .cfi_endproc

; LINUX-X64: test5b:
; LINUX-X64-NOT: callq __stack_chk_fail
; LINUX-X64: .cfi_endproc

; LINUX-KERNEL-X64: test5b:
; LINUX-KERNEL-X64-NOT: callq __stack_chk_fail
; LINUX-KERNEL-X64: .cfi_endproc

; DARWIN-X64: test5b:
; DARWIN-X64-NOT: callq ___stack_chk_fail
; DARWIN-X64: .cfi_endproc
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test5c:  no arrays / no nested arrays
;          sspstrong attribute
; Requires protector.
; FIXME: Once strong heuristic is implemented, this should _not_ require
;        a protector
define void @test5c(i8* %a) nounwind uwtable sspstrong {
entry:
; LINUX-I386: test5c:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test5c:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test5c:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test5c:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

; test5d:  no arrays / no nested arrays
;          sspreq attribute
; Requires protector.
define void @test5d(i8* %a) nounwind uwtable sspreq {
entry:
; LINUX-I386: test5d:
; LINUX-I386: mov{{l|q}} %gs:
; LINUX-I386: calll __stack_chk_fail

; LINUX-X64: test5d:
; LINUX-X64: mov{{l|q}} %fs:
; LINUX-X64: callq __stack_chk_fail

; LINUX-KERNEL-X64: test5d:
; LINUX-KERNEL-X64: mov{{l|q}} %gs:
; LINUX-KERNEL-X64: callq __stack_chk_fail

; DARWIN-X64: test5d:
; DARWIN-X64: mov{{l|q}} ___stack_chk_guard
; DARWIN-X64: callq ___stack_chk_fail
  %a.addr = alloca i8*, align 8
  store i8* %a, i8** %a.addr, align 8
  %0 = load i8** %a.addr, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i8* %0)
  ret void
}

declare i8* @strcpy(i8*, i8*)
declare i32 @printf(i8*, ...)
