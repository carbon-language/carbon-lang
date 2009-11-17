// RUN: clang-cc -analyze -analyzer-experimental-internal-checks -checker-cfref --analyzer-store=region -analyzer-constraints=range -fblocks --analyzer-output=plist -o - %s | FileCheck %s

void test_null_init(void) {
  int *p = 0;
  *p = 0xDEADBEEF;
}
// CHECK: <?xml version="1.0" encoding="UTF-8"?>
// CHECK: <!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
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
// CHECK:      <key>kind</key><string>event</string>
// CHECK:      <key>location</key>
// CHECK:      <dict>
// CHECK:       <key>line</key><integer>4</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>4</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>4</integer>
// CHECK:          <key>col</key><integer>8</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
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
// CHECK:            <key>line</key><integer>4</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>4</integer>
// CHECK:            <key>col</key><integer>8</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:          </array>
// CHECK:         <key>end</key>
// CHECK:          <array>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>5</integer>
// CHECK:            <key>col</key><integer>3</integer>
// CHECK:            <key>file</key><integer>0</integer>
// CHECK:           </dict>
// CHECK:           <dict>
// CHECK:            <key>line</key><integer>5</integer>
// CHECK:            <key>col</key><integer>4</integer>
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
// CHECK:       <key>line</key><integer>5</integer>
// CHECK:       <key>col</key><integer>3</integer>
// CHECK:       <key>file</key><integer>0</integer>
// CHECK:      </dict>
// CHECK:      <key>ranges</key>
// CHECK:      <array>
// CHECK:        <array>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>5</integer>
// CHECK:          <key>col</key><integer>3</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:         <dict>
// CHECK:          <key>line</key><integer>5</integer>
// CHECK:          <key>col</key><integer>4</integer>
// CHECK:          <key>file</key><integer>0</integer>
// CHECK:         </dict>
// CHECK:        </array>
// CHECK:      </array>
// CHECK:      <key>extended_message</key>
// CHECK:      <string>Dereference of null pointer</string>
// CHECK:      <key>message</key>
// CHECK: <string>Dereference of null pointer</string>
// CHECK:     </dict>
// CHECK:    </array>
// CHECK:    <key>description</key><string>Dereference of null pointer</string>
// CHECK:    <key>category</key><string>Logic error</string>
// CHECK:    <key>type</key><string>Null pointer dereference</string>
// CHECK:   <key>location</key>
// CHECK:   <dict>
// CHECK:    <key>line</key><integer>5</integer>
// CHECK:    <key>col</key><integer>3</integer>
// CHECK:    <key>file</key><integer>0</integer>
// CHECK:   </dict>
// CHECK:   </dict>
// CHECK:  </array>
// CHECK: </dict>
// CHECK: </plist>
