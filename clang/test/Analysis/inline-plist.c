// RUN: %clang --analyze %s -Xclang -analyzer-inline-call -o %t
// RUN: FileCheck -input-file %t %s

// <rdar://problem/10967815>
void mmm(int y) {
  if (y != 0)
      y++;
}

int foo(int x, int y) {
    mmm(y);
    if (x != 0)
      x++;
    return 5/x;
}

// CHECK: <?xml version="1.0" encoding="UTF-8"?>
// CHECK: <plist version="1.0">
// CHECK: <dict>
// CHECK:  <key>files</key>
// CHECK:  <array>
// CHECK:  </array>
// CHECK:  <key>diagnostics</key>
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
// CHECK:            <key>line</key><integer>11</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>11</integer>
// CHECK:            <key>col</key><integer>10</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
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
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>9</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>9</integer>
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
// CHECK:       <key>line</key><integer>12</integer>
// CHECK:       <key>col</key><integer>9</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>9</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>12</integer>
// CHECK:          <key>col</key><integer>14</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Assuming &apos;x&apos; is equal to 0</string>
// CHECK:      <key>message</key>
// CHECK: <string>Assuming &apos;x&apos; is equal to 0</string>
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
// CHECK:            <key>col</key><integer>9</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>12</integer>
// CHECK:            <key>col</key><integer>9</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:        </dict>
// CHECK:       </array>
// CHECK:     </dict>
// CHECK:     <dict>
// CHECK:      <key>kind</key><string>control</string>
// CHECK:      <key>edges</key>
// CHECK:       <array>
// CHECK:        <dict>
// CHECK:         <key>start</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>5</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>12</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>14</integer>
// CHECK:            <key>col</key><integer>12</integer>
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
// CHECK:       <key>line</key><integer>14</integer>
// CHECK:       <key>col</key><integer>12</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>14</integer>
// CHECK:          <key>col</key><integer>12</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>14</integer>
// CHECK:          <key>col</key><integer>14</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Division by zero</string>
// CHECK:      <key>message</key>
// CHECK: <string>Division by zero</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Division by zero</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Division by zero</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>14</integer>
// CHECK:    <key>col</key><integer>12</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:  </array>
// CHECK: </dict>
// CHECK: </plist>

