//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Base {
public:
  Base () = default;
  virtual int func() { return 1; }
  virtual ~Base() = default;
};

class Derived : public Base {
private:
  int m_derived_data;
public:
  Derived () : Base(), m_derived_data(0x0fedbeef) {}
  virtual ~Derived() = default;
  virtual int func() { return m_derived_data; }
};

int main (int argc, char const *argv[])
{
  Base *base = new Derived();
    return 0; //% stream = lldb.SBStream()
    //% base = self.frame().FindVariable("base")
    //% base.SetPreferDynamicValue(lldb.eDynamicDontRunTarget)
    //% base.GetDescription(stream)
    //% if self.TraceOn(): print(stream.GetData())
    //% self.assertTrue(stream.GetData().startswith("(Derived *"))
}
