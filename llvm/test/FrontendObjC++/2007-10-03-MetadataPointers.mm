// RUN: %llvmgcc -w -x objective-c++ -S %s -o /dev/null

@class NSImage;
void bork() {
  NSImage *nsimage;
  [nsimage release];
}
