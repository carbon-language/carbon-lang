; RUN: llc < %s -march=avr -mcpu=atmega328p | FileCheck %s

; This test checks a basic 'blinking led' program.
; It is written for the ATmega328P

; Derived from the following C program (with some cleanups):
; #include <avr/io.h>
;
; void setup_ddr() {
;   DDRB |= _BV(PB5);
; }
;
; void turn_on() {
;   PORTB |= _BV(PB5);
; }
;
; void turn_off() {
;   PORTB &= ~_BV(PB5);
; }
;
; int main() {
;   setup_ddr();
;
;   while(1) {
;     turn_on();
;     turn_off();
;   }
;
;   return 0;
; }

; Sets up the data direction register.
; CHECK-LABEL: setup_ddr
define void @setup_ddr() {
entry:

  ; This should set the 5th bit of DDRB.
  ; CHECK:      sbi	4, 5
  ; CHECK-NEXT: ret

  %0 = load volatile i8, i8* inttoptr (i16 36 to i8*), align 1
  %conv = zext i8 %0 to i16
  %or = or i16 %conv, 32
  %conv1 = trunc i16 %or to i8
  store volatile i8 %conv1, i8* inttoptr (i16 36 to i8*), align 1
  ret void
}

; Turns on the LED.
; CHECK-LABEL: turn_on
define void @turn_on() {
entry:

  ; This should set the 5th bit of PORTB
  ; CHECK:      sbi	5, 5
  ; CHECK-NEXT: ret

  %0 = load volatile i8, i8* inttoptr (i16 37 to i8*), align 1
  %conv = zext i8 %0 to i16
  %or = or i16 %conv, 32
  %conv1 = trunc i16 %or to i8
  store volatile i8 %conv1, i8* inttoptr (i16 37 to i8*), align 1
  ret void
}

; Turns off the LED.
; CHECK-LABEL: turn_off
define void @turn_off() {
entry:

  ; This should clear the 5th bit of PORTB
  ; CHECK:      cbi     5, 5
  ; CHECK-NEXT: ret

  %0 = load volatile i8, i8* inttoptr (i16 37 to i8*), align 1
  %conv = zext i8 %0 to i16
  %and = and i16 %conv, -33
  %conv1 = trunc i16 %and to i8
  store volatile i8 %conv1, i8* inttoptr (i16 37 to i8*), align 1
  ret void
}

; CHECK-LABEL: main
define i16 @main() {
entry:

  ; CHECK: call setup_ddr
  call void @setup_ddr()

  br label %while.body

; CHECK-LABEL: .LBB3_1
while.body:

  ; CHECK: call turn_on
  call void @turn_on()

  ; CHECK-NEXT: call turn_off
  call void @turn_off()

  ; CHECK-NEXT: rjmp .LBB3_1
  br label %while.body
}
