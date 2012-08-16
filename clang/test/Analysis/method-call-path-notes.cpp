// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=inlining -analyzer-output=text -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-ipa=inlining -analyzer-output=plist-multi-file %s -o - | FileCheck %s

// Test warning about null or uninitialized pointer values used as instance member
// calls.
class TestInstanceCall {
public:
  void foo() {}
};

void test_ic() {
  TestInstanceCall *p; // expected-note {{Variable 'p' declared without an initial value}}
  p->foo(); // expected-warning {{Called C++ object pointer is uninitialized}} expected-note {{Called C++ object pointer is uninitialized}}
}

void test_ic_null() {
  TestInstanceCall *p = 0; // expected-note {{Variable 'p' initialized to a null pointer value}}
  p->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note {{Called C++ object pointer is null}}
}

void test_ic_set_to_null() {
  TestInstanceCall *p;
  p = 0; // expected-note {{Null pointer value stored to 'p'}}
  p->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note {{Called C++ object pointer is null}}
}

void test_ic_null(TestInstanceCall *p) {
  if (!p) // expected-note {{Assuming pointer value is null}} expected-note {{Taking true branch}}
    p->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note{{Called C++ object pointer is null}}
}

void test_ic_member_ptr() {
  TestInstanceCall *p = 0; // expected-note {{Variable 'p' initialized to a null pointer value}}
  typedef void (TestInstanceCall::*IC_Ptr)();
  IC_Ptr bar = &TestInstanceCall::foo;
  (p->*bar)(); // expected-warning {{Called C++ object pointer is null}} expected-note{{Called C++ object pointer is null}}
}

void test_cast(const TestInstanceCall *p) {
  if (!p) // expected-note {{Assuming pointer value is null}} expected-note {{Taking true branch}}
    const_cast<TestInstanceCall *>(p)->foo(); // expected-warning {{Called C++ object pointer is null}} expected-note {{Called C++ object pointer is null}}
}

// CHECK: <?xml version="1.0" encoding="UTF-8"?>
// CHECK: <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
// CHECK: <plist version="1.0">
// CHECK: <dict>
// CHECK:  <key>files</key>
// CHECK:  <array>
// CHECK:   <string>{{.*}}method-call-path-notes.cpp</string>
// CHECK:  </array>
// CHECK:  <key>diagnostics</key>
// CHECK:  <array>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>12</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>21</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Variable &apos;p&apos; declared without an initial value</string>
// CHECK:      <key>message</key>
// CHECK: <string>Variable &apos;p&apos; declared without an initial value</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>18</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>13</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>13</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>13</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>13</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>13</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Called C++ object pointer is uninitialized</string>
// CHECK:      <key>message</key>
// CHECK: <string>Called C++ object pointer is uninitialized</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Called C++ object pointer is uninitialized</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Called C++ object pointer is uninitialized</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>test_ic</string>
// CHECK:   <key>issue_hash</key><integer>2</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>13</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>17</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>17</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>17</integer>
// CHECK:          <key>col</key><integer>21</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Variable &apos;p&apos; initialized to a null pointer value</string>
// CHECK:      <key>message</key>
// CHECK: <string>Variable &apos;p&apos; initialized to a null pointer value</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>17</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>17</integer>
// CHECK:            <key>col</key><integer>18</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>18</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>18</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>18</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>18</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>18</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Called C++ object pointer is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Called C++ object pointer is null</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Called C++ object pointer is null</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Called C++ object pointer is null</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>test_ic_null</string>
// CHECK:   <key>issue_hash</key><integer>2</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>18</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>22</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>22</integer>
// CHECK:            <key>col</key><integer>18</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>23</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>23</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>23</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Null pointer value stored to &apos;p&apos;</string>
// CHECK:      <key>message</key>
// CHECK: <string>Null pointer value stored to &apos;p&apos;</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>23</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>24</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>24</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>24</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>24</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>24</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Called C++ object pointer is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Called C++ object pointer is null</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Called C++ object pointer is null</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Called C++ object pointer is null</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>test_ic_set_to_null</string>
// CHECK:   <key>issue_hash</key><integer>3</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>24</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>28</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>28</integer>
// CHECK:            <key>col</key><integer>4</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>28</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>28</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>28</integer>
// CHECK:       <key>col</key><integer>7</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>28</integer>
// CHECK:          <key>col</key><integer>7</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>28</integer>
// CHECK:          <key>col</key><integer>8</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Assuming pointer value is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Assuming pointer value is null</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>28</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>28</integer>
// CHECK:            <key>col</key><integer>7</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>29</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>29</integer>
// CHECK:       <key>col</key><integer>5</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>29</integer>
// CHECK:          <key>col</key><integer>5</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>29</integer>
// CHECK:          <key>col</key><integer>5</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Called C++ object pointer is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Called C++ object pointer is null</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Called C++ object pointer is null</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Called C++ object pointer is null</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>test_ic_null</string>
// CHECK:   <key>issue_hash</key><integer>2</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>29</integer>
// CHECK:    <key>col</key><integer>5</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>33</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>33</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>33</integer>
// CHECK:          <key>col</key><integer>21</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Variable &apos;p&apos; initialized to a null pointer value</string>
// CHECK:      <key>message</key>
// CHECK: <string>Variable &apos;p&apos; initialized to a null pointer value</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>33</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>33</integer>
// CHECK:            <key>col</key><integer>18</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>36</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>36</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>36</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>36</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Called C++ object pointer is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Called C++ object pointer is null</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Called C++ object pointer is null</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Called C++ object pointer is null</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>test_ic_member_ptr</string>
// CHECK:   <key>issue_hash</key><integer>4</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>36</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:   <dict>
// CHECK:    <key>path</key>
// CHECK:    <array>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>40</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>40</integer>
// CHECK:            <key>col</key><integer>4</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>41</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>41</integer>
// CHECK:            <key>col</key><integer>14</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>41</integer>
// CHECK:       <key>col</key><integer>5</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>41</integer>
// CHECK:          <key>col</key><integer>5</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>41</integer>
// CHECK:          <key>col</key><integer>37</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Called C++ object pointer is null</string>
// CHECK:      <key>message</key>
// CHECK: <string>Called C++ object pointer is null</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Called C++ object pointer is null</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Called C++ object pointer is null</string>
// CHECK:   <key>issue_context_kind</key><string>function</string>
// CHECK:   <key>issue_context</key><string>test_cast</string>
// CHECK:   <key>issue_hash</key><integer>2</integer>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>41</integer>
// CHECK:    <key>col</key><integer>5</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:  </array>
// CHECK: </dict>
// CHECK: </plist>
