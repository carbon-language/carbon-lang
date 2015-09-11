// RUN: %clang_cc1 -std=c++11 -fsyntax-only -analyze -analyzer-checker=core -analyzer-config inline-lambdas=true -analyzer-output plist -verify %s -o %t
// RUN: FileCheck --input-file=%t %s


// Diagnostic inside a lambda

void diagnosticFromLambda() {
  int i = 0;
  [=] {
    int p = 5/i; // expected-warning{{Division by zero}}
    (void)p;
  }();
}

// CHECK:  <array>
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
// CHECK:            <key>line</key><integer>8</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>8</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>9</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>9</integer>
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
// CHECK:       <key>line</key><integer>9</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>9</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>5</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>The value 0 is assigned to field &apos;&apos;</string>
// CHECK:      <key>message</key>
// CHECK:      <string>The value 0 is assigned to field &apos;&apos;</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>9</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>9</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>5</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>0</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Calling &apos;operator()&apos;</string>
// CHECK:      <key>message</key>
// CHECK:      <string>Calling &apos;operator()&apos;</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>9</integer>
// CHECK:       <key>col</key><integer>5</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>depth</key><integer>1</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Entered call from &apos;diagnosticFromLambda&apos;</string>
// CHECK:      <key>message</key>
// CHECK:      <string>Entered call from &apos;diagnosticFromLambda&apos;</string>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>9</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>9</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>10</integer>
// CHECK:            <key>col</key><integer>14</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>10</integer>
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
// CHECK:       <key>line</key><integer>10</integer>
// CHECK:       <key>col</key><integer>14</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>10</integer>
// CHECK:          <key>col</key><integer>13</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>10</integer>
// CHECK:          <key>col</key><integer>15</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>depth</key><integer>1</integer>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Division by zero</string>
// CHECK:      <key>message</key>
// CHECK:      <string>Division by zero</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Division by zero</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Division by zero</string>
// CHECK:    <key>check_name</key><string>core.DivideZero</string>
// CHECK:   <key>issue_context_kind</key><string>C++ method</string>
// CHECK:   <key>issue_context</key><string>operator()</string>
// CHECK:   <key>issue_hash</key><string>1</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>10</integer>
// CHECK:    <key>col</key><integer>14</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:  </array>

