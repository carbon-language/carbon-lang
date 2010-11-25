// RUN: %llvmgcc -xc++ -S -o - %s | grep _ZN11AccessFlags6strlenEv

struct AccessFlags {
  void strlen();
};

void AccessFlags::strlen() { }

