; RUN: llvm-upgrade < %s | llvm-as | opt -condprop -disable-output
; PR979
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend" ]
	%struct.IO_APIC_reg_00 = type { uint }
	%struct.Qdisc = type { int (%struct.sk_buff*, %struct.Qdisc*)*, %struct.sk_buff* (%struct.Qdisc*)*, uint, %struct.Qdisc_ops*, %struct.Qdisc*, uint, %struct.bluez_skb_cb, %struct.sk_buff_head, %struct.net_device*, %struct.tc_stats, int (%struct.sk_buff*, %struct.Qdisc*)*, %struct.Qdisc*, [1 x sbyte] }
	%struct.Qdisc_class_ops = type { int (%struct.Qdisc*, uint, %struct.Qdisc*, %struct.Qdisc**)*, %struct.Qdisc* (%struct.Qdisc*, uint)*, uint (%struct.Qdisc*, uint)*, void (%struct.Qdisc*, uint)*, int (%struct.Qdisc*, uint, uint, %struct._agp_version**, uint*)*, int (%struct.Qdisc*, uint)*, void (%struct.Qdisc*, %struct.qdisc_walker*)*, %struct.tcf_proto** (%struct.Qdisc*, uint)*, uint (%struct.Qdisc*, uint, uint)*, void (%struct.Qdisc*, uint)*, int (%struct.Qdisc*, uint, %struct.sk_buff*, %struct.tcmsg*)* }
	%struct.Qdisc_ops = type { %struct.Qdisc_ops*, %struct.Qdisc_class_ops*, [16 x sbyte], int, int (%struct.sk_buff*, %struct.Qdisc*)*, %struct.sk_buff* (%struct.Qdisc*)*, int (%struct.sk_buff*, %struct.Qdisc*)*, uint (%struct.Qdisc*)*, int (%struct.Qdisc*, %struct._agp_version*)*, void (%struct.Qdisc*)*, void (%struct.Qdisc*)*, int (%struct.Qdisc*, %struct._agp_version*)*, int (%struct.Qdisc*, %struct.sk_buff*)* }
	%struct.ViceFid = type { uint, uint, uint }
	%struct.__wait_queue = type { uint, %struct.task_struct*, %struct.list_head }
	%struct.__wait_queue_head = type { %struct.IO_APIC_reg_00, %struct.list_head }
	%struct._agp_version = type { ushort, ushort }
	%struct._drm_i810_overlay_t = type { uint, uint }
	%struct.address_space = type { %struct.list_head, %struct.list_head, %struct.list_head, uint, %struct.address_space_operations*, %struct.inode*, %struct.vm_area_struct*, %struct.vm_area_struct*, %struct.IO_APIC_reg_00, int }
	%struct.address_space_operations = type { int (%struct.page*)*, int (%struct.file*, %struct.page*)*, int (%struct.page*)*, int (%struct.file*, %struct.page*, uint, uint)*, int (%struct.file*, %struct.page*, uint, uint)*, int (%struct.address_space*, int)*, int (%struct.page*, uint)*, int (%struct.page*, int)*, int (int, %struct.inode*, %struct.kiobuf*, uint, int)*, int (int, %struct.file*, %struct.kiobuf*, uint, int)*, void (%struct.page*)* }
	%struct.audio_buf_info = type { int, int, int, int }
	%struct.autofs_packet_hdr = type { int, int }
	%struct.block_device = type { %struct.list_head, %struct.bluez_skb_cb, %struct.inode*, ushort, int, %struct.block_device_operations*, %struct.semaphore, %struct.list_head }
	%struct.block_device_operations = type { int (%struct.inode*, %struct.file*)*, int (%struct.inode*, %struct.file*)*, int (%struct.inode*, %struct.file*, uint, uint)*, int (ushort)*, int (ushort)*, %struct.module* }
	%struct.bluez_skb_cb = type { int }
	%struct.buffer_head = type { %struct.buffer_head*, uint, ushort, ushort, ushort, %struct.bluez_skb_cb, ushort, uint, uint, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head**, sbyte*, %struct.page*, void (%struct.buffer_head*, int)*, sbyte*, uint, %struct.__wait_queue_head, %struct.list_head }
	%struct.char_device = type { %struct.list_head, %struct.bluez_skb_cb, ushort, %struct.bluez_skb_cb, %struct.semaphore }
	%struct.completion = type { uint, %struct.__wait_queue_head }
	%struct.cramfs_info = type { uint, uint, uint, uint }
	%struct.dentry = type { %struct.bluez_skb_cb, uint, %struct.inode*, %struct.dentry*, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, int, %struct.qstr, uint, %struct.dentry_operations*, %struct.super_block*, uint, sbyte*, [16 x ubyte] }
	%struct.dentry_operations = type { int (%struct.dentry*, int)*, int (%struct.dentry*, %struct.qstr*)*, int (%struct.dentry*, %struct.qstr*, %struct.qstr*)*, int (%struct.dentry*)*, void (%struct.dentry*)*, void (%struct.dentry*, %struct.inode*)* }
	%struct.dev_mc_list = type { %struct.dev_mc_list*, [8 x ubyte], ubyte, int, int }
	%struct.dnotify_struct = type { %struct.dnotify_struct*, uint, int, %struct.file*, %struct.files_struct* }
	%struct.dquot = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.__wait_queue_head, %struct.__wait_queue_head, int, int, %struct.super_block*, uint, ushort, long, short, short, uint, %struct.mem_dqblk }
	%struct.dquot_operations = type { void (%struct.inode*, int)*, void (%struct.inode*)*, int (%struct.inode*, ulong, int)*, int (%struct.inode*, uint)*, void (%struct.inode*, ulong)*, void (%struct.inode*, uint)*, int (%struct.inode*, %struct.iattr*)*, int (%struct.dquot*)* }
	%struct.drm_clip_rect = type { ushort, ushort, ushort, ushort }
	%struct.drm_ctx_priv_map = type { uint, sbyte* }
	%struct.drm_mga_indices = type { int, uint, uint, int }
	%struct.dst_entry = type { %struct.dst_entry*, %struct.bluez_skb_cb, int, %struct.net_device*, int, int, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, int, %struct.neighbour*, %struct.hh_cache*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)*, %struct.dst_ops*, [0 x sbyte] }
	%struct.dst_ops = type { ushort, ushort, uint, int ()*, %struct.dst_entry* (%struct.dst_entry*, uint)*, %struct.dst_entry* (%struct.dst_entry*, %struct.sk_buff*)*, void (%struct.dst_entry*)*, %struct.dst_entry* (%struct.dst_entry*)*, void (%struct.sk_buff*)*, int, %struct.bluez_skb_cb, %struct.kmem_cache_s* }
	%struct.e820entry = type { ulong, ulong, uint }
	%struct.exec_domain = type { sbyte*, void (int, %struct.pt_regs*)*, ubyte, ubyte, uint*, uint*, %struct.map_segment*, %struct.map_segment*, %struct.map_segment*, %struct.map_segment*, %struct.module*, %struct.exec_domain* }
	%struct.ext2_inode_info = type { [15 x uint], uint, uint, ubyte, ubyte, uint, uint, uint, uint, uint, uint, uint, uint, uint, int }
	%struct.ext3_inode_info = type { [15 x uint], uint, uint, uint, uint, uint, uint, uint, uint, uint, %struct.list_head, long, %struct.rw_semaphore }
	%struct.fasync_struct = type { int, int, %struct.fasync_struct*, %struct.file* }
	%struct.file = type { %struct.list_head, %struct.dentry*, %struct.vfsmount*, %struct.file_operations*, %struct.bluez_skb_cb, uint, ushort, long, uint, uint, uint, uint, uint, %struct.drm_mga_indices, uint, uint, int, uint, sbyte*, %struct.kiobuf*, int }
	%struct.file_lock = type { %struct.file_lock*, %struct.list_head, %struct.list_head, %struct.files_struct*, uint, %struct.__wait_queue_head, %struct.file*, ubyte, ubyte, long, long, void (%struct.file_lock*)*, void (%struct.file_lock*)*, void (%struct.file_lock*)*, %struct.fasync_struct*, uint, { %struct.nfs_lock_info } }
	%struct.file_operations = type { %struct.module*, long (%struct.file*, long, int)*, int (%struct.file*, sbyte*, uint, long*)*, int (%struct.file*, sbyte*, uint, long*)*, int (%struct.file*, sbyte*, int (sbyte*, sbyte*, int, long, uint, uint)*)*, uint (%struct.file*, %struct.poll_table_struct*)*, int (%struct.inode*, %struct.file*, uint, uint)*, int (%struct.file*, %struct.vm_area_struct*)*, int (%struct.inode*, %struct.file*)*, int (%struct.file*)*, int (%struct.inode*, %struct.file*)*, int (%struct.file*, %struct.dentry*, int)*, int (int, %struct.file*, int)*, int (%struct.file*, int, %struct.file_lock*)*, int (%struct.file*, %struct.iovec*, uint, long*)*, int (%struct.file*, %struct.iovec*, uint, long*)*, int (%struct.file*, %struct.page*, int, uint, long*, int)*, uint (%struct.file*, uint, uint, uint, uint)* }
	%struct.file_system_type = type { sbyte*, int, %struct.super_block* (%struct.super_block*, sbyte*, int)*, %struct.module*, %struct.file_system_type*, %struct.list_head }
	%struct.files_struct = type { %struct.bluez_skb_cb, %typedef.rwlock_t, int, int, int, %struct.file**, %typedef.__kernel_fd_set*, %typedef.__kernel_fd_set*, %typedef.__kernel_fd_set, %typedef.__kernel_fd_set, [32 x %struct.file*] }
	%struct.fs_disk_quota = type { sbyte, sbyte, ushort, uint, ulong, ulong, ulong, ulong, ulong, ulong, int, int, ushort, ushort, int, ulong, ulong, ulong, int, ushort, short, [8 x sbyte] }
	%struct.fs_quota_stat = type { sbyte, ushort, sbyte, %struct.e820entry, %struct.e820entry, uint, int, int, int, ushort, ushort }
	%struct.fs_struct = type { %struct.bluez_skb_cb, %typedef.rwlock_t, int, %struct.dentry*, %struct.dentry*, %struct.dentry*, %struct.vfsmount*, %struct.vfsmount*, %struct.vfsmount* }
	%struct.hh_cache = type { %struct.hh_cache*, %struct.bluez_skb_cb, ushort, int, int (%struct.sk_buff*)*, %typedef.rwlock_t, [32 x uint] }
	%struct.i387_fxsave_struct = type { ushort, ushort, ushort, ushort, int, int, int, int, int, int, [32 x int], [32 x int], [56 x int] }
	%struct.iattr = type { uint, ushort, uint, uint, long, int, int, int, uint }
	%struct.if_dqblk = type { ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, uint }
	%struct.if_dqinfo = type { ulong, ulong, uint, uint }
	%struct.ifmap = type { uint, uint, ushort, ubyte, ubyte, ubyte }
	%struct.ifreq = type { { [16 x sbyte] }, %typedef.dvd_authinfo }
	%struct.inode = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, uint, %struct.bluez_skb_cb, ushort, ushort, ushort, uint, uint, ushort, long, int, int, int, uint, uint, uint, uint, ushort, %struct.semaphore, %struct.rw_semaphore, %struct.semaphore, %struct.inode_operations*, %struct.file_operations*, %struct.super_block*, %struct.__wait_queue_head, %struct.file_lock*, %struct.address_space*, %struct.address_space, [2 x %struct.dquot*], %struct.list_head, %struct.pipe_inode_info*, %struct.block_device*, %struct.char_device*, uint, %struct.dnotify_struct*, uint, uint, ubyte, %struct.bluez_skb_cb, uint, uint, { %struct.ext2_inode_info, %struct.ext3_inode_info, %struct.msdos_inode_info, %struct.iso_inode_info, %struct.nfs_inode_info, %struct._drm_i810_overlay_t, %struct.shmem_inode_info, %struct.proc_inode_info, %struct.socket, %struct.usbdev_inode_info, sbyte* } }
	%struct.inode_operations = type { int (%struct.inode*, %struct.dentry*, int)*, %struct.dentry* (%struct.inode*, %struct.dentry*)*, int (%struct.dentry*, %struct.inode*, %struct.dentry*)*, int (%struct.inode*, %struct.dentry*)*, int (%struct.inode*, %struct.dentry*, sbyte*)*, int (%struct.inode*, %struct.dentry*, int)*, int (%struct.inode*, %struct.dentry*)*, int (%struct.inode*, %struct.dentry*, int, int)*, int (%struct.inode*, %struct.dentry*, %struct.inode*, %struct.dentry*)*, int (%struct.dentry*, sbyte*, int)*, int (%struct.dentry*, %struct.nameidata*)*, void (%struct.inode*)*, int (%struct.inode*, int)*, int (%struct.dentry*)*, int (%struct.dentry*, %struct.iattr*)*, int (%struct.dentry*, %struct.iattr*)*, int (%struct.dentry*, sbyte*, sbyte*, uint, int)*, int (%struct.dentry*, sbyte*, sbyte*, uint)*, int (%struct.dentry*, sbyte*, uint)*, int (%struct.dentry*, sbyte*)* }
	%struct.iovec = type { sbyte*, uint }
	%struct.ip_options = type { uint, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, [0 x ubyte] }
	%struct.isapnp_dma = type { ubyte, ubyte, %struct.isapnp_resources*, %struct.isapnp_dma* }
	%struct.isapnp_irq = type { ushort, ubyte, ubyte, %struct.isapnp_resources*, %struct.isapnp_irq* }
	%struct.isapnp_mem = type { uint, uint, uint, uint, ubyte, ubyte, %struct.isapnp_resources*, %struct.isapnp_mem* }
	%struct.isapnp_mem32 = type { [17 x ubyte], %struct.isapnp_resources*, %struct.isapnp_mem32* }
	%struct.isapnp_port = type { ushort, ushort, ubyte, ubyte, ubyte, ubyte, %struct.isapnp_resources*, %struct.isapnp_port* }
	%struct.isapnp_resources = type { ushort, ushort, %struct.isapnp_port*, %struct.isapnp_irq*, %struct.isapnp_dma*, %struct.isapnp_mem*, %struct.isapnp_mem32*, %struct.pci_dev*, %struct.isapnp_resources*, %struct.isapnp_resources* }
	%struct.iso_inode_info = type { uint, ubyte, [3 x ubyte], uint, int }
	%struct.iw_handler_def = type opaque
	%struct.iw_statistics = type opaque
	%struct.k_sigaction = type { %struct.sigaction }
	%struct.kern_ipc_perm = type { int, uint, uint, uint, uint, ushort, uint }
	%struct.kiobuf = type { int, int, int, int, uint, %struct.page**, %struct.buffer_head**, uint*, %struct.bluez_skb_cb, int, void (%struct.kiobuf*)*, %struct.__wait_queue_head }
	%struct.kmem_cache_s = type { %struct.list_head, %struct.list_head, %struct.list_head, uint, uint, uint, %struct.IO_APIC_reg_00, uint, uint, uint, uint, uint, uint, %struct.kmem_cache_s*, uint, uint, void (sbyte*, %struct.kmem_cache_s*, uint)*, void (sbyte*, %struct.kmem_cache_s*, uint)*, uint, [20 x sbyte], %struct.list_head, [32 x %struct._drm_i810_overlay_t*], uint }
	%struct.linux_binfmt = type { %struct.linux_binfmt*, %struct.module*, int (%struct.linux_binprm*, %struct.pt_regs*)*, int (%struct.file*)*, int (int, %struct.pt_regs*, %struct.file*)*, uint, int (%struct.linux_binprm*, sbyte*)* }
	%struct.linux_binprm = type { [128 x sbyte], [32 x %struct.page*], uint, int, %struct.file*, int, int, uint, uint, uint, int, int, sbyte*, uint, uint }
	%struct.list_head = type { %struct.list_head*, %struct.list_head* }
	%struct.llva_sigcontext = type { %typedef.llva_icontext_t, %typedef.llva_fp_state_t, uint, uint, uint, uint, [1 x uint], sbyte* }
	%struct.map_segment = type opaque
	%struct.mem_dqblk = type { uint, uint, ulong, uint, uint, uint, int, int }
	%struct.mem_dqinfo = type { %struct.quota_format_type*, int, uint, uint, { %struct.ViceFid } }
	%struct.mm_struct = type { %struct.vm_area_struct*, %struct.rb_root_s, %struct.vm_area_struct*, %struct.IO_APIC_reg_00*, %struct.bluez_skb_cb, %struct.bluez_skb_cb, int, %struct.rw_semaphore, %struct.IO_APIC_reg_00, %struct.list_head, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, %struct.iovec }
	%struct.module = type { uint, %struct.module*, sbyte*, uint, %struct.bluez_skb_cb, uint, uint, uint, %struct.drm_ctx_priv_map*, %struct.module_ref*, %struct.module_ref*, int ()*, void ()*, %struct._drm_i810_overlay_t*, %struct._drm_i810_overlay_t*, %struct.module_persist*, %struct.module_persist*, int ()*, int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte* }
	%struct.module_persist = type opaque
	%struct.module_ref = type { %struct.module*, %struct.module*, %struct.module_ref* }
	%struct.msdos_inode_info = type { uint, int, int, int, int, int, %struct.inode*, %struct.list_head }
	%struct.msghdr = type { sbyte*, int, %struct.iovec*, uint, sbyte*, uint, uint }
	%struct.msq_setbuf = type { uint, uint, uint, ushort }
	%struct.nameidata = type { %struct.dentry*, %struct.vfsmount*, %struct.qstr, uint, int }
	%struct.namespace = type { %struct.bluez_skb_cb, %struct.vfsmount*, %struct.list_head, %struct.rw_semaphore }
	%struct.neigh_ops = type { int, void (%struct.neighbour*)*, void (%struct.neighbour*, %struct.sk_buff*)*, void (%struct.neighbour*, %struct.sk_buff*)*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)* }
	%struct.neigh_parms = type { %struct.neigh_parms*, int (%struct.neighbour*)*, %struct.neigh_table*, int, sbyte*, sbyte*, int, int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.neigh_table = type { %struct.neigh_table*, int, int, int, uint (sbyte*, %struct.net_device*)*, int (%struct.neighbour*)*, int (%struct.pneigh_entry*)*, void (%struct.pneigh_entry*)*, void (%struct.sk_buff*)*, sbyte*, %struct.neigh_parms, int, int, int, int, uint, %struct.timer_list, %struct.timer_list, %struct.sk_buff_head, int, %typedef.rwlock_t, uint, %struct.neigh_parms*, %struct.kmem_cache_s*, %struct.tasklet_struct, %struct.cramfs_info, [32 x %struct.neighbour*], [16 x %struct.pneigh_entry*] }
	%struct.neighbour = type { %struct.neighbour*, %struct.neigh_table*, %struct.neigh_parms*, %struct.net_device*, uint, uint, uint, ubyte, ubyte, ubyte, ubyte, %struct.bluez_skb_cb, %typedef.rwlock_t, [8 x ubyte], %struct.hh_cache*, %struct.bluez_skb_cb, int (%struct.sk_buff*)*, %struct.sk_buff_head, %struct.timer_list, %struct.neigh_ops*, [0 x ubyte] }
	%struct.net_bridge_port = type opaque
	%struct.net_device = type { [16 x sbyte], uint, uint, uint, uint, uint, uint, ubyte, ubyte, uint, %struct.net_device*, int (%struct.net_device*)*, %struct.net_device*, int, int, %struct.net_device_stats* (%struct.net_device*)*, %struct.iw_statistics* (%struct.net_device*)*, %struct.iw_handler_def*, uint, uint, ushort, ushort, ushort, ushort, uint, ushort, ushort, sbyte*, %struct.net_device*, [8 x ubyte], [8 x ubyte], ubyte, %struct.dev_mc_list*, int, int, int, int, %struct.timer_list, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct.list_head, int, int, %struct.Qdisc*, %struct.Qdisc*, %struct.Qdisc*, %struct.Qdisc*, uint, %struct.IO_APIC_reg_00, int, %struct.IO_APIC_reg_00, %struct.bluez_skb_cb, int, int, void (%struct.net_device*)*, void (%struct.net_device*)*, int (%struct.net_device*)*, int (%struct.net_device*)*, int (%struct.sk_buff*, %struct.net_device*)*, int (%struct.net_device*, int*)*, int (%struct.sk_buff*, %struct.net_device*, ushort, sbyte*, sbyte*, uint)*, int (%struct.sk_buff*)*, void (%struct.net_device*)*, int (%struct.net_device*, sbyte*)*, int (%struct.net_device*, %struct.ifreq*, int)*, int (%struct.net_device*, %struct.ifmap*)*, int (%struct.neighbour*, %struct.hh_cache*)*, void (%struct.hh_cache*, %struct.net_device*, ubyte*)*, int (%struct.net_device*, int)*, void (%struct.net_device*)*, void (%struct.net_device*, %struct.vlan_group*)*, void (%struct.net_device*, ushort)*, void (%struct.net_device*, ushort)*, int (%struct.sk_buff*, ubyte*)*, int (%struct.net_device*, %struct.neigh_parms*)*, int (%struct.net_device*, %struct.dst_entry*)*, %struct.module*, %struct.net_bridge_port* }
	%struct.net_device_stats = type { uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint }
	%struct.nf_conntrack = type { %struct.bluez_skb_cb, void (%struct.nf_conntrack*)* }
	%struct.nf_ct_info = type { %struct.nf_conntrack* }
	%struct.nfs_fh = type { ushort, [64 x ubyte] }
	%struct.nfs_inode_info = type { ulong, %struct.nfs_fh, ushort, uint, ulong, ulong, ulong, uint, uint, uint, [2 x uint], %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, uint, uint, uint, uint, %struct.rpc_cred* }
	%struct.nfs_lock_info = type { uint, uint, %struct.nlm_host* }
	%struct.nlm_host = type opaque
	%struct.open_request = type { %struct.open_request*, uint, uint, ushort, ushort, ubyte, ubyte, ushort, uint, uint, uint, uint, %struct.or_calltable*, %struct.sock*, { %struct.tcp_v4_open_req } }
	%struct.or_calltable = type { int, int (%struct.sock*, %struct.open_request*, %struct.dst_entry*)*, void (%struct.sk_buff*, %struct.open_request*)*, void (%struct.open_request*)*, void (%struct.sk_buff*)* }
	%struct.page = type { %struct.list_head, %struct.address_space*, uint, %struct.page*, %struct.bluez_skb_cb, uint, %struct.list_head, %struct.page**, %struct.buffer_head* }
	%struct.pci_bus = type { %struct.list_head, %struct.pci_bus*, %struct.list_head, %struct.list_head, %struct.pci_dev*, [4 x %struct.resource*], %struct.pci_ops*, sbyte*, %struct.proc_dir_entry*, ubyte, ubyte, ubyte, ubyte, [48 x sbyte], ushort, ushort, uint, ubyte, ubyte, ubyte, ubyte }
	%struct.pci_dev = type { %struct.list_head, %struct.list_head, %struct.pci_bus*, %struct.pci_bus*, sbyte*, %struct.proc_dir_entry*, uint, ushort, ushort, ushort, ushort, uint, ubyte, ubyte, %struct.pci_driver*, sbyte*, ulong, uint, [4 x ushort], [4 x ushort], uint, [12 x %struct.resource], [2 x %struct.resource], [2 x %struct.resource], [90 x sbyte], [8 x sbyte], int, int, ushort, ushort, int (%struct.pci_dev*)*, int (%struct.pci_dev*)*, int (%struct.pci_dev*)* }
	%struct.pci_device_id = type { uint, uint, uint, uint, uint, uint, uint }
	%struct.pci_driver = type { %struct.list_head, sbyte*, %struct.pci_device_id*, int (%struct.pci_dev*, %struct.pci_device_id*)*, void (%struct.pci_dev*)*, int (%struct.pci_dev*, uint)*, int (%struct.pci_dev*, uint)*, int (%struct.pci_dev*)*, int (%struct.pci_dev*, uint, int)* }
	%struct.pci_ops = type { int (%struct.pci_dev*, int, ubyte*)*, int (%struct.pci_dev*, int, ushort*)*, int (%struct.pci_dev*, int, uint*)*, int (%struct.pci_dev*, int, ubyte)*, int (%struct.pci_dev*, int, ushort)*, int (%struct.pci_dev*, int, uint)* }
	%struct.pipe_inode_info = type { %struct.__wait_queue_head, sbyte*, uint, uint, uint, uint, uint, uint, uint, uint }
	%struct.pneigh_entry = type { %struct.pneigh_entry*, %struct.net_device*, [0 x ubyte] }
	%struct.poll_table_entry = type { %struct.file*, %struct.__wait_queue, %struct.__wait_queue_head* }
	%struct.poll_table_page = type { %struct.poll_table_page*, %struct.poll_table_entry*, [0 x %struct.poll_table_entry] }
	%struct.poll_table_struct = type { int, %struct.poll_table_page* }
	%struct.proc_dir_entry = type { ushort, ushort, sbyte*, ushort, ushort, uint, uint, uint, %struct.inode_operations*, %struct.file_operations*, int (sbyte*, sbyte**, int, int)*, %struct.module*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, sbyte*, int (sbyte*, sbyte**, int, int, int*, sbyte*)*, int (%struct.file*, sbyte*, uint, sbyte*)*, %struct.bluez_skb_cb, int, ushort }
	%struct.proc_inode_info = type { %struct.task_struct*, int, { int (%struct.task_struct*, sbyte*)* }, %struct.file* }
	%struct.proto = type { void (%struct.sock*, int)*, int (%struct.sock*, %struct.sockaddr*, int)*, int (%struct.sock*, int)*, %struct.sock* (%struct.sock*, int, int*)*, int (%struct.sock*, int, uint)*, int (%struct.sock*)*, int (%struct.sock*)*, void (%struct.sock*, int)*, int (%struct.sock*, int, int, sbyte*, int)*, int (%struct.sock*, int, int, sbyte*, int*)*, int (%struct.sock*, %struct.msghdr*, int)*, int (%struct.sock*, %struct.msghdr*, int, int, int, int*)*, int (%struct.sock*, %struct.sockaddr*, int)*, int (%struct.sock*, %struct.sk_buff*)*, void (%struct.sock*)*, void (%struct.sock*)*, int (%struct.sock*, ushort)*, [32 x sbyte], [32 x { int, [28 x ubyte] }] }
	%struct.proto_ops = type { int, int (%struct.socket*)*, int (%struct.socket*, %struct.sockaddr*, int)*, int (%struct.socket*, %struct.sockaddr*, int, int)*, int (%struct.socket*, %struct.socket*)*, int (%struct.socket*, %struct.socket*, int)*, int (%struct.socket*, %struct.sockaddr*, int*, int)*, uint (%struct.file*, %struct.socket*, %struct.poll_table_struct*)*, int (%struct.socket*, uint, uint)*, int (%struct.socket*, int)*, int (%struct.socket*, int)*, int (%struct.socket*, int, int, sbyte*, int)*, int (%struct.socket*, int, int, sbyte*, int*)*, int (%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)*, int (%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)*, int (%struct.file*, %struct.socket*, %struct.vm_area_struct*)*, int (%struct.socket*, %struct.page*, int, uint, int)* }
	%struct.pt_regs = type { int, int, int, int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.qdisc_walker = type { int, int, int, int (%struct.Qdisc*, uint, %struct.qdisc_walker*)* }
	%struct.qstr = type { ubyte*, uint, uint }
	%struct.quota_format_ops = type { int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.dquot*)*, int (%struct.dquot*)* }
	%struct.quota_format_type = type { int, %struct.quota_format_ops*, %struct.module*, %struct.quota_format_type* }
	%struct.quota_info = type { uint, %struct.semaphore, %struct.semaphore, [2 x %struct.file*], [2 x %struct.mem_dqinfo], [2 x %struct.quota_format_ops*] }
	%struct.quotactl_ops = type { int (%struct.super_block*, int, int, sbyte*)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int, %struct.if_dqinfo*)*, int (%struct.super_block*, int, %struct.if_dqinfo*)*, int (%struct.super_block*, int, uint, %struct.if_dqblk*)*, int (%struct.super_block*, int, uint, %struct.if_dqblk*)*, int (%struct.super_block*, %struct.fs_quota_stat*)*, int (%struct.super_block*, uint, int)*, int (%struct.super_block*, int, uint, %struct.fs_disk_quota*)*, int (%struct.super_block*, int, uint, %struct.fs_disk_quota*)* }
	%struct.rb_node_s = type { %struct.rb_node_s*, int, %struct.rb_node_s*, %struct.rb_node_s* }
	%struct.rb_root_s = type { %struct.rb_node_s* }
	%struct.resource = type { sbyte*, uint, uint, uint, %struct.resource*, %struct.resource*, %struct.resource* }
	%struct.revectored_struct = type { [8 x uint] }
	%struct.rpc_auth = type { [8 x %struct.rpc_cred*], uint, uint, uint, uint, uint, %struct.rpc_authops* }
	%struct.rpc_authops = type { uint, sbyte*, %struct.rpc_auth* (%struct.rpc_clnt*)*, void (%struct.rpc_auth*)*, %struct.rpc_cred* (int)* }
	%struct.rpc_clnt = type { %struct.bluez_skb_cb, %struct.rpc_xprt*, %struct.rpc_procinfo*, uint, sbyte*, sbyte*, %struct.rpc_auth*, %struct.rpc_stat*, uint, uint, uint, %struct.rpc_rtt, %struct.msq_setbuf, %struct.rpc_wait_queue, int, [32 x sbyte] }
	%struct.rpc_cred = type { %struct.rpc_cred*, %struct.rpc_auth*, %struct.rpc_credops*, uint, %struct.bluez_skb_cb, ushort, uint, uint }
	%struct.rpc_credops = type { void (%struct.rpc_cred*)*, int (%struct.rpc_cred*, int)*, uint* (%struct.rpc_task*, uint*, int)*, int (%struct.rpc_task*)*, uint* (%struct.rpc_task*, uint*)* }
	%struct.rpc_message = type { uint, sbyte*, sbyte*, %struct.rpc_cred* }
	%struct.rpc_procinfo = type { sbyte*, int (sbyte*, uint*, sbyte*)*, int (sbyte*, uint*, sbyte*)*, uint, uint, uint }
	%struct.rpc_program = type { sbyte*, uint, uint, %struct.rpc_version**, %struct.rpc_stat* }
	%struct.rpc_rqst = type { %struct.rpc_xprt*, %struct.rpc_timeout, %struct.xdr_buf, %struct.xdr_buf, %struct.rpc_task*, uint, %struct.rpc_rqst*, int, int, %struct.list_head, %struct.xdr_buf, [2 x uint], uint, int, int, int }
	%struct.rpc_rtt = type { int, [5 x int], [5 x int], %struct.bluez_skb_cb }
	%struct.rpc_stat = type { %struct.rpc_program*, uint, uint, uint, uint, uint, uint, uint, uint, uint }
	%struct.rpc_task = type { %struct.list_head, uint, %struct.list_head, %struct.rpc_clnt*, %struct.rpc_rqst*, int, %struct.rpc_wait_queue*, %struct.rpc_message, uint*, ubyte, ubyte, ubyte, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, void (%struct.rpc_task*)*, sbyte*, %struct.timer_list, %struct.__wait_queue_head, uint, ushort, ubyte, uint, ushort }
	%struct.rpc_timeout = type { uint, uint, uint, uint, short, ubyte }
	%struct.rpc_version = type { uint, uint, %struct.rpc_procinfo* }
	%struct.rpc_wait_queue = type { %struct.list_head, sbyte* }
	%struct.rpc_xprt = type { %struct.socket*, %struct.sock*, %struct.rpc_timeout, %struct.sockaddr_in, int, uint, uint, uint, uint, %struct.rpc_wait_queue, %struct.rpc_wait_queue, %struct.rpc_wait_queue, %struct.rpc_wait_queue, %struct.rpc_rqst*, [16 x %struct.rpc_rqst], uint, ubyte, uint, uint, uint, uint, uint, uint, %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, %struct.rpc_task*, %struct.list_head, void (%struct.sock*, int)*, void (%struct.sock*)*, void (%struct.sock*)*, %struct.__wait_queue_head }
	%struct.rw_semaphore = type { int, %struct.IO_APIC_reg_00, %struct.list_head }
	%struct.scm_cookie = type { %struct.ViceFid, %struct.scm_fp_list*, uint }
	%struct.scm_fp_list = type { int, [255 x %struct.file*] }
	%struct.sem_array = type { %struct.kern_ipc_perm, int, int, %struct.autofs_packet_hdr*, %struct.sem_queue*, %struct.sem_queue**, %struct.sem_undo*, uint }
	%struct.sem_queue = type { %struct.sem_queue*, %struct.sem_queue**, %struct.task_struct*, %struct.sem_undo*, int, int, %struct.sem_array*, int, %struct.sembuf*, int, int }
	%struct.sem_undo = type { %struct.sem_undo*, %struct.sem_undo*, int, short* }
	%struct.semaphore = type { %struct.bluez_skb_cb, int, %struct.__wait_queue_head }
	%struct.sembuf = type { ushort, short, short }
	%struct.seq_file = type { sbyte*, uint, uint, uint, long, %struct.semaphore, %struct.seq_operations*, sbyte* }
	%struct.seq_operations = type { sbyte* (%struct.seq_file*, long*)*, void (%struct.seq_file*, sbyte*)*, sbyte* (%struct.seq_file*, sbyte*, long*)*, int (%struct.seq_file*, sbyte*)* }
	%struct.shmem_inode_info = type { %struct.IO_APIC_reg_00, uint, [16 x %struct.IO_APIC_reg_00], sbyte**, uint, uint, %struct.list_head, %struct.inode* }
	%struct.sigaction = type { void (int)*, uint, void ()*, %typedef.sigset_t }
	%struct.siginfo = type { int, int, int, { [29 x int] } }
	%struct.signal_struct = type { %struct.bluez_skb_cb, [64 x %struct.k_sigaction], %struct.IO_APIC_reg_00 }
	%struct.sigpending = type { %struct.sigqueue*, %struct.sigqueue**, %typedef.sigset_t }
	%struct.sigqueue = type { %struct.sigqueue*, %struct.siginfo }
	%struct.sk_buff = type { %struct.sk_buff*, %struct.sk_buff*, %struct.sk_buff_head*, %struct.sock*, %struct.autofs_packet_hdr, %struct.net_device*, %struct.net_device*, { ubyte* }, { ubyte* }, { ubyte* }, %struct.dst_entry*, [48 x sbyte], uint, uint, uint, ubyte, ubyte, ubyte, ubyte, uint, %struct.bluez_skb_cb, ushort, ushort, uint, ubyte*, ubyte*, ubyte*, ubyte*, void (%struct.sk_buff*)*, uint, uint, %struct.nf_ct_info*, uint }
	%struct.sk_buff_head = type { %struct.sk_buff*, %struct.sk_buff*, uint, %struct.IO_APIC_reg_00 }
	%struct.sock = type { uint, uint, ushort, ushort, int, %struct.sock*, %struct.sock**, %struct.sock*, %struct.sock**, ubyte, ubyte, ushort, ushort, ubyte, ubyte, %struct.bluez_skb_cb, %typedef.socket_lock_t, int, %struct.__wait_queue_head*, %struct.dst_entry*, %typedef.rwlock_t, %struct.bluez_skb_cb, %struct.sk_buff_head, %struct.bluez_skb_cb, %struct.sk_buff_head, %struct.bluez_skb_cb, int, int, uint, uint, int, %struct.sock*, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, ubyte, ubyte, ubyte, ubyte, int, int, uint, int, %struct.sock*, { %struct.sk_buff*, %struct.sk_buff* }, %typedef.rwlock_t, %struct.sk_buff_head, %struct.proto*, { %struct.tcp_opt }, int, int, ushort, ushort, uint, ushort, ubyte, ubyte, %struct.ViceFid, int, int, int, { %struct.unix_opt }, %struct.timer_list, %struct.autofs_packet_hdr, %struct.socket*, sbyte*, void (%struct.sock*)*, void (%struct.sock*, int)*, void (%struct.sock*)*, void (%struct.sock*)*, int (%struct.sock*, %struct.sk_buff*)*, void (%struct.sock*)* }
	%struct.sockaddr = type { ushort, [14 x sbyte] }
	%struct.sockaddr_in = type { ushort, ushort, %struct.IO_APIC_reg_00, [8 x ubyte] }
	%struct.sockaddr_un = type { ushort, [108 x sbyte] }
	%struct.socket = type { uint, uint, %struct.proto_ops*, %struct.inode*, %struct.fasync_struct*, %struct.file*, %struct.sock*, %struct.__wait_queue_head, short, ubyte }
	%struct.statfs = type { int, int, int, int, int, int, int, %typedef.__kernel_fsid_t, int, [6 x int] }
	%struct.super_block = type { %struct.list_head, ushort, uint, ubyte, ubyte, ulong, %struct.file_system_type*, %struct.super_operations*, %struct.dquot_operations*, %struct.quotactl_ops*, uint, uint, %struct.dentry*, %struct.rw_semaphore, %struct.semaphore, int, %struct.bluez_skb_cb, %struct.list_head, %struct.list_head, %struct.list_head, %struct.block_device*, %struct.list_head, %struct.quota_info, { [115 x uint] }, %struct.semaphore, %struct.semaphore }
	%struct.super_operations = type { %struct.inode* (%struct.super_block*)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.inode*, sbyte*)*, void (%struct.inode*)*, void (%struct.inode*, int)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, int (%struct.super_block*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, int (%struct.super_block*, %struct.statfs*)*, int (%struct.super_block*, int*, sbyte*)*, void (%struct.inode*)*, void (%struct.super_block*)*, %struct.dentry* (%struct.super_block*, uint*, int, int, int)*, int (%struct.dentry*, uint*, int*, int)*, int (%struct.seq_file*, %struct.vfsmount*)* }
	%struct.task_struct = type { int, uint, int, %struct.IO_APIC_reg_00, %struct.exec_domain*, int, uint, int, int, int, uint, %struct.mm_struct*, int, uint, uint, %struct.list_head, uint, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, uint, uint, %struct.linux_binfmt*, int, int, int, uint, int, int, int, int, int, int, int, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.__wait_queue_head, %struct.completion*, uint, uint, uint, uint, uint, uint, uint, %struct.timer_list, %struct.audio_buf_info, uint, [32 x int], [32 x int], uint, uint, uint, uint, uint, uint, int, uint, uint, uint, uint, uint, uint, uint, uint, int, [32 x uint], uint, uint, uint, int, %struct.user_struct*, [11 x %struct._drm_i810_overlay_t], ushort, [16 x sbyte], int, int, %struct.tty_struct*, uint, %struct.sem_undo*, %struct.sem_queue*, %struct.thread_struct, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.IO_APIC_reg_00, %struct.signal_struct*, %typedef.sigset_t, %struct.sigpending, uint, uint, int (sbyte*)*, sbyte*, %typedef.sigset_t*, uint, uint, %struct.IO_APIC_reg_00, sbyte*, %struct.llva_sigcontext*, uint, %struct.task_struct*, uint, %typedef.llva_icontext_t, %typedef.llva_fp_state_t, uint*, int, sbyte* }
	%struct.tasklet_struct = type { %struct.tasklet_struct*, uint, %struct.bluez_skb_cb, void (uint)*, uint }
	%struct.tc_stats = type { ulong, uint, uint, uint, uint, uint, uint, uint, %struct.IO_APIC_reg_00* }
	%struct.tcf_proto = type { %struct.tcf_proto*, sbyte*, int (%struct.sk_buff*, %struct.tcf_proto*, %struct._drm_i810_overlay_t*)*, uint, uint, uint, %struct.Qdisc*, sbyte*, %struct.tcf_proto_ops* }
	%struct.tcf_proto_ops = type { %struct.tcf_proto_ops*, [16 x sbyte], int (%struct.sk_buff*, %struct.tcf_proto*, %struct._drm_i810_overlay_t*)*, int (%struct.tcf_proto*)*, void (%struct.tcf_proto*)*, uint (%struct.tcf_proto*, uint)*, void (%struct.tcf_proto*, uint)*, int (%struct.tcf_proto*, uint, uint, %struct._agp_version**, uint*)*, int (%struct.tcf_proto*, uint)*, void (%struct.tcf_proto*, %struct.tcf_walker*)*, int (%struct.tcf_proto*, uint, %struct.sk_buff*, %struct.tcmsg*)* }
	%struct.tcf_walker = type { int, int, int, int (%struct.tcf_proto*, uint, %struct.tcf_walker*)* }
	%struct.tcmsg = type { ubyte, ubyte, ushort, int, uint, uint, uint }
	%struct.tcp_func = type { int (%struct.sk_buff*)*, void (%struct.sock*, %struct.tcphdr*, int, %struct.sk_buff*)*, int (%struct.sock*)*, int (%struct.sock*, %struct.sk_buff*)*, %struct.sock* (%struct.sock*, %struct.sk_buff*, %struct.open_request*, %struct.dst_entry*)*, int (%struct.sock*)*, ushort, int (%struct.sock*, int, int, sbyte*, int)*, int (%struct.sock*, int, int, sbyte*, int*)*, void (%struct.sock*, %struct.sockaddr*)*, int }
	%struct.tcp_listen_opt = type { ubyte, int, int, int, uint, [512 x %struct.open_request*] }
	%struct.tcp_opt = type { int, uint, uint, uint, uint, uint, uint, uint, { ubyte, ubyte, ubyte, ubyte, uint, uint, uint, ushort, ushort }, { %struct.sk_buff_head, %struct.task_struct*, %struct.iovec*, int, int }, uint, uint, uint, uint, ushort, ushort, ushort, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, ushort, ushort, uint, uint, uint, %struct.timer_list, %struct.timer_list, %struct.sk_buff_head, %struct.tcp_func*, %struct.sk_buff*, %struct.page*, uint, uint, uint, uint, uint, uint, sbyte, sbyte, sbyte, sbyte, ubyte, ubyte, ubyte, ubyte, uint, uint, uint, int, ushort, ubyte, ubyte, [1 x %struct._drm_i810_overlay_t], [4 x %struct._drm_i810_overlay_t], uint, uint, ubyte, ubyte, ushort, ubyte, ubyte, ushort, uint, uint, uint, uint, uint, uint, int, uint, ushort, ubyte, ubyte, uint, %typedef.rwlock_t, %struct.tcp_listen_opt*, %struct.open_request*, %struct.open_request*, int, uint, uint, int, int, uint, uint }
	%struct.tcp_v4_open_req = type { uint, uint, %struct.ip_options* }
	%struct.tcphdr = type { ushort, ushort, uint, uint, ushort, ushort, ushort, ushort }
	%struct.termios = type { uint, uint, uint, uint, ubyte, [19 x ubyte] }
	%struct.thread_struct = type { uint, uint, uint, uint, uint, [8 x uint], uint, uint, uint, %union.i387_union, %struct.vm86_struct*, uint, uint, uint, uint, int, [33 x uint] }
	%struct.timer_list = type { %struct.list_head, uint, uint, void (uint)* }
	%struct.tq_struct = type { %struct.list_head, uint, void (sbyte*)*, sbyte* }
	%struct.tty_driver = type { int, sbyte*, sbyte*, int, short, short, short, short, short, %struct.termios, int, int*, %struct.proc_dir_entry*, %struct.tty_driver*, %struct.tty_struct**, %struct.termios**, %struct.termios**, sbyte*, int (%struct.tty_struct*, %struct.file*)*, void (%struct.tty_struct*, %struct.file*)*, int (%struct.tty_struct*, int, ubyte*, int)*, void (%struct.tty_struct*, ubyte)*, void (%struct.tty_struct*)*, int (%struct.tty_struct*)*, int (%struct.tty_struct*)*, int (%struct.tty_struct*, %struct.file*, uint, uint)*, void (%struct.tty_struct*, %struct.termios*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, int)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, int)*, void (%struct.tty_struct*, sbyte)*, int (sbyte*, sbyte**, int, int, int*, sbyte*)*, int (%struct.file*, sbyte*, uint, sbyte*)*, %struct.tty_driver*, %struct.tty_driver* }
	%struct.tty_flip_buffer = type { %struct.tq_struct, %struct.semaphore, sbyte*, ubyte*, int, int, [1024 x ubyte], [1024 x sbyte], [4 x ubyte] }
	%struct.tty_ldisc = type { int, sbyte*, int, int, int (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, int (%struct.tty_struct*)*, int (%struct.tty_struct*, %struct.file*, ubyte*, uint)*, int (%struct.tty_struct*, %struct.file*, ubyte*, uint)*, int (%struct.tty_struct*, %struct.file*, uint, uint)*, void (%struct.tty_struct*, %struct.termios*)*, uint (%struct.tty_struct*, %struct.file*, %struct.poll_table_struct*)*, void (%struct.tty_struct*, ubyte*, sbyte*, int)*, int (%struct.tty_struct*)*, void (%struct.tty_struct*)* }
	%struct.tty_struct = type { int, %struct.tty_driver, %struct.tty_ldisc, %struct.termios*, %struct.termios*, int, int, ushort, uint, int, %struct.drm_clip_rect, ubyte, ubyte, %struct.tty_struct*, %struct.fasync_struct*, %struct.tty_flip_buffer, int, int, %struct.__wait_queue_head, %struct.__wait_queue_head, %struct.tq_struct, sbyte*, sbyte*, %struct.list_head, uint, ubyte, ushort, uint, int, [8 x uint], sbyte*, int, int, int, [128 x uint], int, uint, uint, %struct.semaphore, %struct.semaphore, %struct.IO_APIC_reg_00, %struct.tq_struct }
	%struct.unix_address = type { %struct.bluez_skb_cb, int, uint, [0 x %struct.sockaddr_un] }
	%struct.unix_opt = type { %struct.unix_address*, %struct.dentry*, %struct.vfsmount*, %struct.semaphore, %struct.sock*, %struct.sock**, %struct.sock*, %struct.bluez_skb_cb, %typedef.rwlock_t, %struct.__wait_queue_head }
	%struct.usb_bus = type opaque
	%struct.usbdev_inode_info = type { %struct.list_head, %struct.list_head, { %struct.usb_bus* } }
	%struct.user_struct = type { %struct.bluez_skb_cb, %struct.bluez_skb_cb, %struct.bluez_skb_cb, %struct.user_struct*, %struct.user_struct**, uint }
	%struct.vfsmount = type { %struct.list_head, %struct.vfsmount*, %struct.dentry*, %struct.dentry*, %struct.super_block*, %struct.list_head, %struct.list_head, %struct.bluez_skb_cb, int, sbyte*, %struct.list_head }
	%struct.vlan_group = type opaque
	%struct.vm86_regs = type { int, int, int, int, int, int, int, int, int, int, int, int, int, ushort, ushort, int, int, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort }
	%struct.vm86_struct = type { %struct.vm86_regs, uint, uint, uint, %struct.revectored_struct, %struct.revectored_struct }
	%struct.vm_area_struct = type { %struct.mm_struct*, uint, uint, %struct.vm_area_struct*, %struct.IO_APIC_reg_00, uint, %struct.rb_node_s, %struct.vm_area_struct*, %struct.vm_area_struct**, %struct.vm_operations_struct*, uint, %struct.file*, uint, sbyte* }
	%struct.vm_operations_struct = type { void (%struct.vm_area_struct*)*, void (%struct.vm_area_struct*)*, %struct.page* (%struct.vm_area_struct*, uint, int)* }
	%struct.xdr_buf = type { [1 x %struct.iovec], [1 x %struct.iovec], %struct.page**, uint, uint, uint }
	%typedef.__kernel_fd_set = type { [32 x int] }
	%typedef.__kernel_fsid_t = type { [2 x int] }
	%typedef.dvd_authinfo = type { [2 x ulong] }
	%typedef.llva_fp_state_t = type { [7 x uint], [20 x uint] }
	%typedef.llva_icontext_t = type { uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint*, uint }
	%typedef.rwlock_t = type { %struct.IO_APIC_reg_00, %struct.IO_APIC_reg_00, uint }
	%typedef.sigset_t = type { [2 x uint] }
	%typedef.socket_lock_t = type { %struct.IO_APIC_reg_00, uint, %struct.__wait_queue_head }
	%union.i387_union = type { %struct.i387_fxsave_struct }

implementation   ; Functions:

void %rs_init() {
entry:
	br bool false, label %loopentry.0.no_exit.0_crit_edge, label %loopentry.0.loopexit.0_crit_edge

loopentry.0:		; No predecessors!
	unreachable

loopentry.0.loopexit.0_crit_edge:		; preds = %entry
	br label %loopexit.0

loopentry.0.no_exit.0_crit_edge:		; preds = %entry
	br label %no_exit.0

no_exit.0:		; preds = %no_exit.0.no_exit.0_crit_edge, %loopentry.0.no_exit.0_crit_edge
	br bool false, label %no_exit.0.no_exit.0_crit_edge, label %no_exit.0.loopexit.0_crit_edge

no_exit.0.loopexit.0_crit_edge:		; preds = %no_exit.0
	br label %loopexit.0

no_exit.0.no_exit.0_crit_edge:		; preds = %no_exit.0
	br label %no_exit.0

loopexit.0:		; preds = %no_exit.0.loopexit.0_crit_edge, %loopentry.0.loopexit.0_crit_edge
	br bool false, label %then.0, label %loopexit.0.endif.0_crit_edge

loopexit.0.endif.0_crit_edge:		; preds = %loopexit.0
	br label %endif.0

then.0:		; preds = %loopexit.0
	br bool false, label %loopentry.1.no_exit.1_crit_edge, label %loopentry.1.loopexit.1_crit_edge

loopentry.1:		; No predecessors!
	unreachable

loopentry.1.loopexit.1_crit_edge:		; preds = %then.0
	br label %loopexit.1

loopentry.1.no_exit.1_crit_edge:		; preds = %then.0
	br label %no_exit.1

no_exit.1:		; preds = %no_exit.1.backedge, %loopentry.1.no_exit.1_crit_edge
	br bool false, label %shortcirc_next.0, label %no_exit.1.shortcirc_done.0_crit_edge

no_exit.1.shortcirc_done.0_crit_edge:		; preds = %no_exit.1
	br label %shortcirc_done.0

shortcirc_next.0:		; preds = %no_exit.1
	br label %shortcirc_done.0

shortcirc_done.0:		; preds = %shortcirc_next.0, %no_exit.1.shortcirc_done.0_crit_edge
	br bool false, label %then.1, label %endif.1

then.1:		; preds = %shortcirc_done.0
	br bool false, label %then.1.no_exit.1_crit_edge, label %then.1.loopexit.1_crit_edge

then.1.loopexit.1_crit_edge:		; preds = %then.1
	br label %loopexit.1

then.1.no_exit.1_crit_edge:		; preds = %then.1
	br label %no_exit.1.backedge

no_exit.1.backedge:		; preds = %endif.1.no_exit.1_crit_edge, %then.1.no_exit.1_crit_edge
	br label %no_exit.1

endif.1:		; preds = %shortcirc_done.0
	br bool false, label %endif.1.no_exit.1_crit_edge, label %endif.1.loopexit.1_crit_edge

endif.1.loopexit.1_crit_edge:		; preds = %endif.1
	br label %loopexit.1

endif.1.no_exit.1_crit_edge:		; preds = %endif.1
	br label %no_exit.1.backedge

loopexit.1:		; preds = %endif.1.loopexit.1_crit_edge, %then.1.loopexit.1_crit_edge, %loopentry.1.loopexit.1_crit_edge
	br label %endif.0

endif.0:		; preds = %loopexit.1, %loopexit.0.endif.0_crit_edge
	br bool false, label %then.2, label %endif.0.endif.2_crit_edge

endif.0.endif.2_crit_edge:		; preds = %endif.0
	br label %endif.2

then.2:		; preds = %endif.0
	unreachable

dead_block.0:		; No predecessors!
	br label %endif.2

endif.2:		; preds = %dead_block.0, %endif.0.endif.2_crit_edge
	br bool false, label %then.3, label %endif.2.endif.3_crit_edge

endif.2.endif.3_crit_edge:		; preds = %endif.2
	br label %endif.3

then.3:		; preds = %endif.2
	unreachable

dead_block.1:		; No predecessors!
	br label %endif.3

endif.3:		; preds = %dead_block.1, %endif.2.endif.3_crit_edge
	br label %loopentry.2

loopentry.2:		; preds = %endif.6, %endif.3
	br bool false, label %loopentry.2.no_exit.2_crit_edge, label %loopentry.2.loopexit.2_crit_edge

loopentry.2.loopexit.2_crit_edge:		; preds = %loopentry.2
	br label %loopexit.2

loopentry.2.no_exit.2_crit_edge:		; preds = %loopentry.2
	br label %no_exit.2

no_exit.2:		; preds = %then.5.no_exit.2_crit_edge, %loopentry.2.no_exit.2_crit_edge
	br bool false, label %then.4, label %no_exit.2.endif.4_crit_edge

no_exit.2.endif.4_crit_edge:		; preds = %no_exit.2
	br label %endif.4

then.4:		; preds = %no_exit.2
	br label %endif.4

endif.4:		; preds = %then.4, %no_exit.2.endif.4_crit_edge
	br bool false, label %shortcirc_next.1, label %endif.4.shortcirc_done.1_crit_edge

endif.4.shortcirc_done.1_crit_edge:		; preds = %endif.4
	br label %shortcirc_done.1

shortcirc_next.1:		; preds = %endif.4
	br bool false, label %then.i21, label %endif.i

then.i21:		; preds = %shortcirc_next.1
	br label %then.5

then.i21.endif.5_crit_edge:		; No predecessors!
	unreachable

then.i21.then.5_crit_edge:		; No predecessors!
	unreachable

endif.i:		; preds = %shortcirc_next.1
	br label %shortcirc_done.1

__check_region.exit:		; No predecessors!
	unreachable

shortcirc_done.1:		; preds = %endif.i, %endif.4.shortcirc_done.1_crit_edge
	br bool false, label %shortcirc_done.1.then.5_crit_edge, label %shortcirc_done.1.endif.5_crit_edge

shortcirc_done.1.endif.5_crit_edge:		; preds = %shortcirc_done.1
	br label %endif.5

shortcirc_done.1.then.5_crit_edge:		; preds = %shortcirc_done.1
	br label %then.5

then.5:		; preds = %shortcirc_done.1.then.5_crit_edge, %then.i21
	br bool false, label %then.5.no_exit.2_crit_edge, label %then.5.loopexit.2_crit_edge

then.5.loopexit.2_crit_edge:		; preds = %then.5
	br label %loopexit.2

then.5.no_exit.2_crit_edge:		; preds = %then.5
	br label %no_exit.2

dead_block_after_continue.0:		; No predecessors!
	unreachable

endif.5:		; preds = %shortcirc_done.1.endif.5_crit_edge
	br bool false, label %then.6, label %endif.5.endif.6_crit_edge

endif.5.endif.6_crit_edge:		; preds = %endif.5
	br label %endif.6

then.6:		; preds = %endif.5
	br label %endif.6

endif.6:		; preds = %then.6, %endif.5.endif.6_crit_edge
	br label %loopentry.2

loopcont.2:		; No predecessors!
	unreachable

loopexit.2:		; preds = %then.5.loopexit.2_crit_edge, %loopentry.2.loopexit.2_crit_edge
	br label %loopentry.3

loopentry.3:		; preds = %endif.9, %loopexit.2
	br bool false, label %loopentry.3.no_exit.3_crit_edge, label %loopentry.3.loopexit.3_crit_edge

loopentry.3.loopexit.3_crit_edge:		; preds = %loopentry.3
	br label %loopexit.3

loopentry.3.no_exit.3_crit_edge:		; preds = %loopentry.3
	br label %no_exit.3

no_exit.3:		; preds = %then.7.no_exit.3_crit_edge, %loopentry.3.no_exit.3_crit_edge
	br bool false, label %then.7, label %no_exit.3.endif.7_crit_edge

no_exit.3.endif.7_crit_edge:		; preds = %no_exit.3
	br label %endif.7

then.7:		; preds = %no_exit.3
	br bool false, label %then.7.no_exit.3_crit_edge, label %then.7.loopexit.3_crit_edge

then.7.loopexit.3_crit_edge:		; preds = %then.7
	br label %loopexit.3

then.7.no_exit.3_crit_edge:		; preds = %then.7
	br label %no_exit.3

dead_block_after_continue.1:		; No predecessors!
	unreachable

endif.7:		; preds = %no_exit.3.endif.7_crit_edge
	br bool false, label %shortcirc_next.2, label %endif.7.shortcirc_done.2_crit_edge

endif.7.shortcirc_done.2_crit_edge:		; preds = %endif.7
	br label %shortcirc_done.2

shortcirc_next.2:		; preds = %endif.7
	br label %shortcirc_done.2

shortcirc_done.2:		; preds = %shortcirc_next.2, %endif.7.shortcirc_done.2_crit_edge
	br bool false, label %shortcirc_next.3, label %shortcirc_done.2.shortcirc_done.3_crit_edge

shortcirc_done.2.shortcirc_done.3_crit_edge:		; preds = %shortcirc_done.2
	br label %shortcirc_done.3

shortcirc_next.3:		; preds = %shortcirc_done.2
	br bool false, label %shortcirc_next.3.shortcirc_done.4_crit_edge, label %shortcirc_next.4

shortcirc_next.3.shortcirc_done.4_crit_edge:		; preds = %shortcirc_next.3
	br label %shortcirc_done.4

shortcirc_next.4:		; preds = %shortcirc_next.3
	br label %shortcirc_done.4

shortcirc_done.4:		; preds = %shortcirc_next.4, %shortcirc_next.3.shortcirc_done.4_crit_edge
	br label %shortcirc_done.3

shortcirc_done.3:		; preds = %shortcirc_done.4, %shortcirc_done.2.shortcirc_done.3_crit_edge
	br bool false, label %then.8, label %shortcirc_done.3.endif.8_crit_edge

shortcirc_done.3.endif.8_crit_edge:		; preds = %shortcirc_done.3
	br label %endif.8

then.8:		; preds = %shortcirc_done.3
	br label %endif.8

endif.8:		; preds = %then.8, %shortcirc_done.3.endif.8_crit_edge
	br bool false, label %then.9, label %else

then.9:		; preds = %endif.8
	br bool false, label %cond_true.0, label %cond_false.0

cond_true.0:		; preds = %then.9
	br label %cond_continue.0

cond_false.0:		; preds = %then.9
	br label %cond_continue.0

cond_continue.0:		; preds = %cond_false.0, %cond_true.0
	br label %endif.9

else:		; preds = %endif.8
	br bool false, label %cond_true.1, label %cond_false.1

cond_true.1:		; preds = %else
	br label %cond_continue.1

cond_false.1:		; preds = %else
	br label %cond_continue.1

cond_continue.1:		; preds = %cond_false.1, %cond_true.1
	br label %endif.9

endif.9:		; preds = %cond_continue.1, %cond_continue.0
	br label %loopentry.3

loopcont.3:		; No predecessors!
	unreachable

loopexit.3:		; preds = %then.7.loopexit.3_crit_edge, %loopentry.3.loopexit.3_crit_edge
	br bool false, label %loopentry.i.i.i2.no_exit.i.i.i4_crit_edge, label %loopentry.i.i.i2.pci_register_driver.exit.i.i_crit_edge

loopentry.i.i.i2:		; No predecessors!
	unreachable

loopentry.i.i.i2.pci_register_driver.exit.i.i_crit_edge:		; preds = %loopexit.3
	br label %pci_register_driver.exit.i.i

loopentry.i.i.i2.no_exit.i.i.i4_crit_edge:		; preds = %loopexit.3
	br label %no_exit.i.i.i4

no_exit.i.i.i4:		; preds = %endif.i.i.i10.no_exit.i.i.i4_crit_edge, %loopentry.i.i.i2.no_exit.i.i.i4_crit_edge
	br bool false, label %then.i.i.i6, label %no_exit.i.i.i4.endif.i.i.i10_crit_edge

no_exit.i.i.i4.endif.i.i.i10_crit_edge:		; preds = %no_exit.i.i.i4
	br label %endif.i.i.i10

then.i.i.i6:		; preds = %no_exit.i.i.i4
	br bool false, label %then.0.i.i.i.i, label %else.i.i.i.i

then.0.i.i.i.i:		; preds = %then.i.i.i6
	br bool false, label %then.1.i.i.i.i, label %endif.1.i.i.i.i

then.1.i.i.i.i:		; preds = %then.0.i.i.i.i
	br label %endif.i.i.i10

endif.1.i.i.i.i:		; preds = %then.0.i.i.i.i
	br bool false, label %endif.1.i.i.i.i.then.i.i.i.i.i.i_crit_edge, label %endif.1.i.i.i.i.endif.i.i.i.i.i.i_crit_edge

endif.1.i.i.i.i.endif.i.i.i.i.i.i_crit_edge:		; preds = %endif.1.i.i.i.i
	br label %endif.i.i.i.i.i.i

endif.1.i.i.i.i.then.i.i.i.i.i.i_crit_edge:		; preds = %endif.1.i.i.i.i
	br label %then.i.i.i.i.i.i

else.i.i.i.i:		; preds = %then.i.i.i6
	br bool false, label %endif.0.i.i.i.i.then.i.i.i.i.i.i_crit_edge, label %endif.0.i.i.i.i.endif.i.i.i.i.i.i_crit_edge

endif.0.i.i.i.i:		; No predecessors!
	unreachable

endif.0.i.i.i.i.endif.i.i.i.i.i.i_crit_edge:		; preds = %else.i.i.i.i
	br label %endif.i.i.i.i.i.i

endif.0.i.i.i.i.then.i.i.i.i.i.i_crit_edge:		; preds = %else.i.i.i.i
	br label %then.i.i.i.i.i.i

then.i.i.i.i.i.i:		; preds = %endif.0.i.i.i.i.then.i.i.i.i.i.i_crit_edge, %endif.1.i.i.i.i.then.i.i.i.i.i.i_crit_edge
	br bool false, label %then.i.i.i.i.i.i.then.2.i.i.i.i_crit_edge, label %then.i.i.i.i.i.i.endif.2.i.i.i.i_crit_edge

then.i.i.i.i.i.i.endif.2.i.i.i.i_crit_edge:		; preds = %then.i.i.i.i.i.i
	br label %endif.2.i.i.i.i

then.i.i.i.i.i.i.then.2.i.i.i.i_crit_edge:		; preds = %then.i.i.i.i.i.i
	br label %then.2.i.i.i.i

endif.i.i.i.i.i.i:		; preds = %endif.0.i.i.i.i.endif.i.i.i.i.i.i_crit_edge, %endif.1.i.i.i.i.endif.i.i.i.i.i.i_crit_edge
	br bool false, label %dev_probe_lock.exit.i.i.i.i.then.2.i.i.i.i_crit_edge, label %dev_probe_lock.exit.i.i.i.i.endif.2.i.i.i.i_crit_edge

dev_probe_lock.exit.i.i.i.i:		; No predecessors!
	unreachable

dev_probe_lock.exit.i.i.i.i.endif.2.i.i.i.i_crit_edge:		; preds = %endif.i.i.i.i.i.i
	br label %endif.2.i.i.i.i

dev_probe_lock.exit.i.i.i.i.then.2.i.i.i.i_crit_edge:		; preds = %endif.i.i.i.i.i.i
	br label %then.2.i.i.i.i

then.2.i.i.i.i:		; preds = %dev_probe_lock.exit.i.i.i.i.then.2.i.i.i.i_crit_edge, %then.i.i.i.i.i.i.then.2.i.i.i.i_crit_edge
	br label %endif.2.i.i.i.i

endif.2.i.i.i.i:		; preds = %then.2.i.i.i.i, %dev_probe_lock.exit.i.i.i.i.endif.2.i.i.i.i_crit_edge, %then.i.i.i.i.i.i.endif.2.i.i.i.i_crit_edge
	br bool false, label %then.i.i2.i.i.i.i, label %endif.i.i3.i.i.i.i

then.i.i2.i.i.i.i:		; preds = %endif.2.i.i.i.i
	br label %endif.i.i.i10

endif.i.i3.i.i.i.i:		; preds = %endif.2.i.i.i.i
	br label %endif.i.i.i10

dev_probe_unlock.exit.i.i.i.i:		; No predecessors!
	unreachable

pci_announce_device.exit.i.i.i:		; No predecessors!
	unreachable

endif.i.i.i10:		; preds = %endif.i.i3.i.i.i.i, %then.i.i2.i.i.i.i, %then.1.i.i.i.i, %no_exit.i.i.i4.endif.i.i.i10_crit_edge
	br bool false, label %endif.i.i.i10.no_exit.i.i.i4_crit_edge, label %endif.i.i.i10.pci_register_driver.exit.i.i_crit_edge

endif.i.i.i10.pci_register_driver.exit.i.i_crit_edge:		; preds = %endif.i.i.i10
	br label %pci_register_driver.exit.i.i

endif.i.i.i10.no_exit.i.i.i4_crit_edge:		; preds = %endif.i.i.i10
	br label %no_exit.i.i.i4

pci_register_driver.exit.i.i:		; preds = %endif.i.i.i10.pci_register_driver.exit.i.i_crit_edge, %loopentry.i.i.i2.pci_register_driver.exit.i.i_crit_edge
	br bool false, label %then.0.i.i12, label %endif.0.i.i13

then.0.i.i12:		; preds = %pci_register_driver.exit.i.i
	br label %probe_serial_pci.exit

then.0.i.i12.probe_serial_pci.exit_crit_edge:		; No predecessors!
	unreachable

then.0.i.i12.then.i_crit_edge:		; No predecessors!
	br label %then.i

endif.0.i.i13:		; preds = %pci_register_driver.exit.i.i
	br bool false, label %then.1.i.i14, label %endif.0.i.i13.endif.1.i.i15_crit_edge

endif.0.i.i13.endif.1.i.i15_crit_edge:		; preds = %endif.0.i.i13
	br label %endif.1.i.i15

then.1.i.i14:		; preds = %endif.0.i.i13
	br label %endif.1.i.i15

endif.1.i.i15:		; preds = %then.1.i.i14, %endif.0.i.i13.endif.1.i.i15_crit_edge
	br bool false, label %loopentry.i8.i.i.no_exit.i9.i.i_crit_edge, label %loopentry.i8.i.i.pci_unregister_driver.exit.i.i_crit_edge

loopentry.i8.i.i:		; No predecessors!
	unreachable

loopentry.i8.i.i.pci_unregister_driver.exit.i.i_crit_edge:		; preds = %endif.1.i.i15
	br label %pci_unregister_driver.exit.i.i

loopentry.i8.i.i.no_exit.i9.i.i_crit_edge:		; preds = %endif.1.i.i15
	br label %no_exit.i9.i.i

no_exit.i9.i.i:		; preds = %endif.0.i.i.i.no_exit.i9.i.i_crit_edge, %loopentry.i8.i.i.no_exit.i9.i.i_crit_edge
	br bool false, label %then.0.i.i.i, label %no_exit.i9.i.i.endif.0.i.i.i_crit_edge

no_exit.i9.i.i.endif.0.i.i.i_crit_edge:		; preds = %no_exit.i9.i.i
	br label %endif.0.i.i.i

then.0.i.i.i:		; preds = %no_exit.i9.i.i
	br bool false, label %then.1.i.i.i, label %then.0.i.i.i.endif.1.i.i.i_crit_edge

then.0.i.i.i.endif.1.i.i.i_crit_edge:		; preds = %then.0.i.i.i
	br label %endif.1.i.i.i

then.1.i.i.i:		; preds = %then.0.i.i.i
	br label %endif.1.i.i.i

endif.1.i.i.i:		; preds = %then.1.i.i.i, %then.0.i.i.i.endif.1.i.i.i_crit_edge
	br label %endif.0.i.i.i

endif.0.i.i.i:		; preds = %endif.1.i.i.i, %no_exit.i9.i.i.endif.0.i.i.i_crit_edge
	br bool false, label %endif.0.i.i.i.no_exit.i9.i.i_crit_edge, label %endif.0.i.i.i.pci_unregister_driver.exit.i.i_crit_edge

endif.0.i.i.i.pci_unregister_driver.exit.i.i_crit_edge:		; preds = %endif.0.i.i.i
	br label %pci_unregister_driver.exit.i.i

endif.0.i.i.i.no_exit.i9.i.i_crit_edge:		; preds = %endif.0.i.i.i
	br label %no_exit.i9.i.i

pci_unregister_driver.exit.i.i:		; preds = %endif.0.i.i.i.pci_unregister_driver.exit.i.i_crit_edge, %loopentry.i8.i.i.pci_unregister_driver.exit.i.i_crit_edge
	br bool false, label %pci_module_init.exit.i.then.i_crit_edge, label %pci_module_init.exit.i.probe_serial_pci.exit_crit_edge

pci_module_init.exit.i:		; No predecessors!
	unreachable

pci_module_init.exit.i.probe_serial_pci.exit_crit_edge:		; preds = %pci_unregister_driver.exit.i.i
	br label %probe_serial_pci.exit

pci_module_init.exit.i.then.i_crit_edge:		; preds = %pci_unregister_driver.exit.i.i
	br label %then.i

then.i:		; preds = %pci_module_init.exit.i.then.i_crit_edge, %then.0.i.i12.then.i_crit_edge
	br label %probe_serial_pci.exit

probe_serial_pci.exit:		; preds = %then.i, %pci_module_init.exit.i.probe_serial_pci.exit_crit_edge, %then.0.i.i12
	br bool false, label %then.0.i, label %endif.0.i

then.0.i:		; preds = %probe_serial_pci.exit
	ret void

endif.0.i:		; preds = %probe_serial_pci.exit
	br bool false, label %loopentry.0.i.no_exit.0.i_crit_edge, label %loopentry.0.i.loopexit.0.i_crit_edge

loopentry.0.i:		; No predecessors!
	unreachable

loopentry.0.i.loopexit.0.i_crit_edge:		; preds = %endif.0.i
	br label %loopexit.0.i

loopentry.0.i.no_exit.0.i_crit_edge:		; preds = %endif.0.i
	br label %no_exit.0.i

no_exit.0.i:		; preds = %loopcont.0.i.no_exit.0.i_crit_edge, %loopentry.0.i.no_exit.0.i_crit_edge
	br bool false, label %then.1.i, label %endif.1.i

then.1.i:		; preds = %no_exit.0.i
	br label %loopcont.0.i

endif.1.i:		; preds = %no_exit.0.i
	br bool false, label %loopentry.1.i.no_exit.1.i_crit_edge, label %loopentry.1.i.loopexit.1.i_crit_edge

loopentry.1.i:		; No predecessors!
	unreachable

loopentry.1.i.loopexit.1.i_crit_edge:		; preds = %endif.1.i
	br label %loopexit.1.i

loopentry.1.i.no_exit.1.i_crit_edge:		; preds = %endif.1.i
	br label %no_exit.1.i

no_exit.1.i:		; preds = %endif.2.i.no_exit.1.i_crit_edge, %loopentry.1.i.no_exit.1.i_crit_edge
	br bool false, label %shortcirc_next.0.i, label %no_exit.1.i.shortcirc_done.0.i_crit_edge

no_exit.1.i.shortcirc_done.0.i_crit_edge:		; preds = %no_exit.1.i
	br label %shortcirc_done.0.i

shortcirc_next.0.i:		; preds = %no_exit.1.i
	br label %shortcirc_done.0.i

shortcirc_done.0.i:		; preds = %shortcirc_next.0.i, %no_exit.1.i.shortcirc_done.0.i_crit_edge
	br bool false, label %then.2.i, label %endif.2.i

then.2.i:		; preds = %shortcirc_done.0.i
	br bool false, label %then.2.i.then.3.i_crit_edge, label %then.2.i.else.i_crit_edge

then.2.i.else.i_crit_edge:		; preds = %then.2.i
	br label %else.i

then.2.i.then.3.i_crit_edge:		; preds = %then.2.i
	br label %then.3.i

endif.2.i:		; preds = %shortcirc_done.0.i
	br bool false, label %endif.2.i.no_exit.1.i_crit_edge, label %endif.2.i.loopexit.1.i_crit_edge

endif.2.i.loopexit.1.i_crit_edge:		; preds = %endif.2.i
	br label %loopexit.1.i

endif.2.i.no_exit.1.i_crit_edge:		; preds = %endif.2.i
	br label %no_exit.1.i

loopexit.1.i:		; preds = %endif.2.i.loopexit.1.i_crit_edge, %loopentry.1.i.loopexit.1.i_crit_edge
	br bool false, label %loopexit.1.i.then.3.i_crit_edge, label %loopexit.1.i.else.i_crit_edge

loopexit.1.i.else.i_crit_edge:		; preds = %loopexit.1.i
	br label %else.i

loopexit.1.i.then.3.i_crit_edge:		; preds = %loopexit.1.i
	br label %then.3.i

then.3.i:		; preds = %loopexit.1.i.then.3.i_crit_edge, %then.2.i.then.3.i_crit_edge
	br bool false, label %shortcirc_next.1.i, label %then.3.i.shortcirc_done.1.i_crit_edge

then.3.i.shortcirc_done.1.i_crit_edge:		; preds = %then.3.i
	br label %shortcirc_done.1.i

shortcirc_next.1.i:		; preds = %then.3.i
	br label %shortcirc_done.1.i

shortcirc_done.1.i:		; preds = %shortcirc_next.1.i, %then.3.i.shortcirc_done.1.i_crit_edge
	br bool false, label %then.4.i, label %endif.4.i

then.4.i:		; preds = %shortcirc_done.1.i
	br label %endif.3.i

endif.4.i:		; preds = %shortcirc_done.1.i
	br label %endif.3.i

else.i:		; preds = %loopexit.1.i.else.i_crit_edge, %then.2.i.else.i_crit_edge
	br bool false, label %shortcirc_next.0.i.i, label %else.i.shortcirc_done.0.i.i_crit_edge

else.i.shortcirc_done.0.i.i_crit_edge:		; preds = %else.i
	br label %shortcirc_done.0.i.i

shortcirc_next.0.i.i:		; preds = %else.i
	br label %shortcirc_done.0.i.i

shortcirc_done.0.i.i:		; preds = %shortcirc_next.0.i.i, %else.i.shortcirc_done.0.i.i_crit_edge
	br bool false, label %shortcirc_next.1.i.i, label %shortcirc_done.0.i.i.shortcirc_done.1.i.i_crit_edge

shortcirc_done.0.i.i.shortcirc_done.1.i.i_crit_edge:		; preds = %shortcirc_done.0.i.i
	br label %shortcirc_done.1.i.i

shortcirc_next.1.i.i:		; preds = %shortcirc_done.0.i.i
	br bool false, label %loopentry.i.i2.i.no_exit.i.i3.i_crit_edge, label %loopentry.i.i2.i.loopexit.i.i.i_crit_edge

loopentry.i.i2.i:		; No predecessors!
	unreachable

loopentry.i.i2.i.loopexit.i.i.i_crit_edge:		; preds = %shortcirc_next.1.i.i
	br label %loopexit.i.i.i

loopentry.i.i2.i.no_exit.i.i3.i_crit_edge:		; preds = %shortcirc_next.1.i.i
	br label %no_exit.i.i3.i

no_exit.i.i3.i:		; preds = %endif.i.i.i.no_exit.i.i3.i_crit_edge, %loopentry.i.i2.i.no_exit.i.i3.i_crit_edge
	br bool false, label %shortcirc_next.0.i.i.i, label %no_exit.i.i3.i.shortcirc_done.0.i.i.i_crit_edge

no_exit.i.i3.i.shortcirc_done.0.i.i.i_crit_edge:		; preds = %no_exit.i.i3.i
	br label %shortcirc_done.0.i.i.i

shortcirc_next.0.i.i.i:		; preds = %no_exit.i.i3.i
	br label %shortcirc_done.0.i.i.i

shortcirc_done.0.i.i.i:		; preds = %shortcirc_next.0.i.i.i, %no_exit.i.i3.i.shortcirc_done.0.i.i.i_crit_edge
	br bool false, label %shortcirc_next.1.i.i.i, label %shortcirc_done.0.i.i.i.shortcirc_done.1.i.i.i_crit_edge

shortcirc_done.0.i.i.i.shortcirc_done.1.i.i.i_crit_edge:		; preds = %shortcirc_done.0.i.i.i
	br label %shortcirc_done.1.i.i.i

shortcirc_next.1.i.i.i:		; preds = %shortcirc_done.0.i.i.i
	br label %shortcirc_done.1.i.i.i

shortcirc_done.1.i.i.i:		; preds = %shortcirc_next.1.i.i.i, %shortcirc_done.0.i.i.i.shortcirc_done.1.i.i.i_crit_edge
	br bool false, label %then.i.i.i, label %endif.i.i.i

then.i.i.i:		; preds = %shortcirc_done.1.i.i.i
	br label %then.0.i.i

then.i.i.i.endif.0.i.i_crit_edge:		; No predecessors!
	unreachable

then.i.i.i.then.0.i.i_crit_edge:		; No predecessors!
	unreachable

endif.i.i.i:		; preds = %shortcirc_done.1.i.i.i
	br bool false, label %endif.i.i.i.no_exit.i.i3.i_crit_edge, label %endif.i.i.i.loopexit.i.i.i_crit_edge

endif.i.i.i.loopexit.i.i.i_crit_edge:		; preds = %endif.i.i.i
	br label %loopexit.i.i.i

endif.i.i.i.no_exit.i.i3.i_crit_edge:		; preds = %endif.i.i.i
	br label %no_exit.i.i3.i

loopexit.i.i.i:		; preds = %endif.i.i.i.loopexit.i.i.i_crit_edge, %loopentry.i.i2.i.loopexit.i.i.i_crit_edge
	br label %shortcirc_done.1.i.i

check_compatible_id.exit.i.i:		; No predecessors!
	unreachable

shortcirc_done.1.i.i:		; preds = %loopexit.i.i.i, %shortcirc_done.0.i.i.shortcirc_done.1.i.i_crit_edge
	br bool false, label %shortcirc_done.1.i.i.then.0.i.i_crit_edge, label %shortcirc_done.1.i.i.endif.0.i.i_crit_edge

shortcirc_done.1.i.i.endif.0.i.i_crit_edge:		; preds = %shortcirc_done.1.i.i
	br label %endif.0.i.i

shortcirc_done.1.i.i.then.0.i.i_crit_edge:		; preds = %shortcirc_done.1.i.i
	br label %then.0.i.i

then.0.i.i:		; preds = %shortcirc_done.1.i.i.then.0.i.i_crit_edge, %then.i.i.i
	br label %then.5.i

then.0.i.i.endif.5.i_crit_edge:		; No predecessors!
	unreachable

then.0.i.i.then.5.i_crit_edge:		; No predecessors!
	unreachable

endif.0.i.i:		; preds = %shortcirc_done.1.i.i.endif.0.i.i_crit_edge
	br bool false, label %endif.0.i.i.shortcirc_done.2.i.i_crit_edge, label %shortcirc_next.2.i.i

endif.0.i.i.shortcirc_done.2.i.i_crit_edge:		; preds = %endif.0.i.i
	br label %shortcirc_done.2.i.i

shortcirc_next.2.i.i:		; preds = %endif.0.i.i
	br label %shortcirc_done.2.i.i

shortcirc_done.2.i.i:		; preds = %shortcirc_next.2.i.i, %endif.0.i.i.shortcirc_done.2.i.i_crit_edge
	br bool false, label %then.1.i.i, label %endif.1.i.i

then.1.i.i:		; preds = %shortcirc_done.2.i.i
	br label %then.5.i

then.1.i.i.endif.5.i_crit_edge:		; No predecessors!
	unreachable

then.1.i.i.then.5.i_crit_edge:		; No predecessors!
	unreachable

endif.1.i.i:		; preds = %shortcirc_done.2.i.i
	br bool false, label %loopentry.0.i7.i.no_exit.0.i8.i_crit_edge, label %loopentry.0.i7.i.loopexit.0.i11.i_crit_edge

loopentry.0.i7.i:		; No predecessors!
	unreachable

loopentry.0.i7.i.loopexit.0.i11.i_crit_edge:		; preds = %endif.1.i.i
	br label %loopexit.0.i11.i

loopentry.0.i7.i.no_exit.0.i8.i_crit_edge:		; preds = %endif.1.i.i
	br label %no_exit.0.i8.i

no_exit.0.i8.i:		; preds = %loopexit.1.i.i.no_exit.0.i8.i_crit_edge, %loopentry.0.i7.i.no_exit.0.i8.i_crit_edge
	br bool false, label %loopentry.1.i9.i.no_exit.1.i10.i_crit_edge, label %loopentry.1.i9.i.loopexit.1.i.i_crit_edge

loopentry.1.i9.i:		; No predecessors!
	unreachable

loopentry.1.i9.i.loopexit.1.i.i_crit_edge:		; preds = %no_exit.0.i8.i
	br label %loopexit.1.i.i

loopentry.1.i9.i.no_exit.1.i10.i_crit_edge:		; preds = %no_exit.0.i8.i
	br label %no_exit.1.i10.i

no_exit.1.i10.i:		; preds = %endif.2.i.i.no_exit.1.i10.i_crit_edge, %loopentry.1.i9.i.no_exit.1.i10.i_crit_edge
	br bool false, label %shortcirc_next.3.i.i, label %no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge

no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge:		; preds = %no_exit.1.i10.i
	br label %shortcirc_done.3.i.i

shortcirc_next.3.i.i:		; preds = %no_exit.1.i10.i
	br bool false, label %shortcirc_next.3.i.i.shortcirc_done.4.i.i_crit_edge, label %shortcirc_next.4.i.i

shortcirc_next.3.i.i.shortcirc_done.4.i.i_crit_edge:		; preds = %shortcirc_next.3.i.i
	br label %shortcirc_done.4.i.i

shortcirc_next.4.i.i:		; preds = %shortcirc_next.3.i.i
	br label %shortcirc_done.4.i.i

shortcirc_done.4.i.i:		; preds = %shortcirc_next.4.i.i, %shortcirc_next.3.i.i.shortcirc_done.4.i.i_crit_edge
	br bool false, label %shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge, label %shortcirc_next.5.i.i

shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge:		; preds = %shortcirc_done.4.i.i
	br label %shortcirc_done.5.i.i

shortcirc_next.5.i.i:		; preds = %shortcirc_done.4.i.i
	%tmp.68.i.i = seteq ushort 0, 1000		; <bool> [#uses=1]
	br label %shortcirc_done.5.i.i

shortcirc_done.5.i.i:		; preds = %shortcirc_next.5.i.i, %shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge
	%shortcirc_val.4.i.i = phi bool [ true, %shortcirc_done.4.i.i.shortcirc_done.5.i.i_crit_edge ], [ %tmp.68.i.i, %shortcirc_next.5.i.i ]		; <bool> [#uses=1]
	br label %shortcirc_done.3.i.i

shortcirc_done.3.i.i:		; preds = %shortcirc_done.5.i.i, %no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge
	%shortcirc_val.5.i.i = phi bool [ false, %no_exit.1.i10.i.shortcirc_done.3.i.i_crit_edge ], [ %shortcirc_val.4.i.i, %shortcirc_done.5.i.i ]		; <bool> [#uses=1]
	br bool %shortcirc_val.5.i.i, label %then.2.i.i, label %endif.2.i.i

then.2.i.i:		; preds = %shortcirc_done.3.i.i
	%port.2.i.i.8.lcssa20 = phi %struct.isapnp_port* [ null, %shortcirc_done.3.i.i ]		; <%struct.isapnp_port*> [#uses=0]
	br label %endif.5.i

then.2.i.i.endif.5.i_crit_edge:		; No predecessors!
	unreachable

then.2.i.i.then.5.i_crit_edge:		; No predecessors!
	unreachable

endif.2.i.i:		; preds = %shortcirc_done.3.i.i
	br bool false, label %endif.2.i.i.no_exit.1.i10.i_crit_edge, label %endif.2.i.i.loopexit.1.i.i_crit_edge

endif.2.i.i.loopexit.1.i.i_crit_edge:		; preds = %endif.2.i.i
	br label %loopexit.1.i.i

endif.2.i.i.no_exit.1.i10.i_crit_edge:		; preds = %endif.2.i.i
	br label %no_exit.1.i10.i

loopexit.1.i.i:		; preds = %endif.2.i.i.loopexit.1.i.i_crit_edge, %loopentry.1.i9.i.loopexit.1.i.i_crit_edge
	br bool false, label %loopexit.1.i.i.no_exit.0.i8.i_crit_edge, label %loopexit.1.i.i.loopexit.0.i11.i_crit_edge

loopexit.1.i.i.loopexit.0.i11.i_crit_edge:		; preds = %loopexit.1.i.i
	br label %loopexit.0.i11.i

loopexit.1.i.i.no_exit.0.i8.i_crit_edge:		; preds = %loopexit.1.i.i
	br label %no_exit.0.i8.i

loopexit.0.i11.i:		; preds = %loopexit.1.i.i.loopexit.0.i11.i_crit_edge, %loopentry.0.i7.i.loopexit.0.i11.i_crit_edge
	br bool false, label %serial_pnp_guess_board.exit.i.then.5.i_crit_edge, label %serial_pnp_guess_board.exit.i.endif.5.i_crit_edge

serial_pnp_guess_board.exit.i:		; No predecessors!
	unreachable

serial_pnp_guess_board.exit.i.endif.5.i_crit_edge:		; preds = %loopexit.0.i11.i
	br label %endif.5.i

serial_pnp_guess_board.exit.i.then.5.i_crit_edge:		; preds = %loopexit.0.i11.i
	br label %then.5.i

then.5.i:		; preds = %serial_pnp_guess_board.exit.i.then.5.i_crit_edge, %then.1.i.i, %then.0.i.i
	br label %loopcont.0.i

endif.5.i:		; preds = %serial_pnp_guess_board.exit.i.endif.5.i_crit_edge, %then.2.i.i
	br label %endif.3.i

endif.3.i:		; preds = %endif.5.i, %endif.4.i, %then.4.i
	br bool false, label %then.6.i, label %endif.3.i.endif.6.i_crit_edge

endif.3.i.endif.6.i_crit_edge:		; preds = %endif.3.i
	br label %endif.6.i

then.6.i:		; preds = %endif.3.i
	br label %loopentry.0.i.i

loopentry.0.i.i:		; preds = %endif.i.i, %then.6.i
	br bool false, label %loopentry.0.i.i.no_exit.0.i.i_crit_edge, label %loopentry.0.i.i.loopexit.0.i.i_crit_edge

loopentry.0.i.i.loopexit.0.i.i_crit_edge:		; preds = %loopentry.0.i.i
	br label %loopexit.0.i.i

loopentry.0.i.i.no_exit.0.i.i_crit_edge:		; preds = %loopentry.0.i.i
	br label %no_exit.0.i.i

no_exit.0.i.i:		; preds = %clear_bit195.exit.i.i.no_exit.0.i.i_crit_edge, %loopentry.0.i.i.no_exit.0.i.i_crit_edge
	br bool false, label %then.i.i, label %endif.i.i

then.i.i:		; preds = %no_exit.0.i.i
	br label %loopentry.i.i.i

loopentry.i.i.i:		; preds = %no_exit.i.i.i, %then.i.i
	br bool false, label %no_exit.i.i.i, label %clear_bit195.exit.i.i

no_exit.i.i.i:		; preds = %loopentry.i.i.i
	br label %loopentry.i.i.i

clear_bit195.exit.i.i:		; preds = %loopentry.i.i.i
	br bool false, label %clear_bit195.exit.i.i.no_exit.0.i.i_crit_edge, label %clear_bit195.exit.i.i.loopexit.0.i.i_crit_edge

clear_bit195.exit.i.i.loopexit.0.i.i_crit_edge:		; preds = %clear_bit195.exit.i.i
	br label %loopexit.0.i.i

clear_bit195.exit.i.i.no_exit.0.i.i_crit_edge:		; preds = %clear_bit195.exit.i.i
	br label %no_exit.0.i.i

endif.i.i:		; preds = %no_exit.0.i.i
	br label %loopentry.0.i.i

loopexit.0.i.i:		; preds = %clear_bit195.exit.i.i.loopexit.0.i.i_crit_edge, %loopentry.0.i.i.loopexit.0.i.i_crit_edge
	br bool false, label %loopentry.1.i.i.no_exit.1.i.i_crit_edge, label %loopentry.1.i.i.avoid_irq_share.exit.i_crit_edge

loopentry.1.i.i:		; No predecessors!
	unreachable

loopentry.1.i.i.avoid_irq_share.exit.i_crit_edge:		; preds = %loopexit.0.i.i
	br label %avoid_irq_share.exit.i

loopentry.1.i.i.no_exit.1.i.i_crit_edge:		; preds = %loopexit.0.i.i
	br label %no_exit.1.i.i

no_exit.1.i.i:		; preds = %loopexit.2.i.i.no_exit.1.i.i_crit_edge, %loopentry.1.i.i.no_exit.1.i.i_crit_edge
	br bool false, label %loopentry.2.i.i.no_exit.2.i.i_crit_edge, label %loopentry.2.i.i.loopexit.2.i.i_crit_edge

loopentry.2.i.i:		; No predecessors!
	unreachable

loopentry.2.i.i.loopexit.2.i.i_crit_edge:		; preds = %no_exit.1.i.i
	br label %loopexit.2.i.i

loopentry.2.i.i.no_exit.2.i.i_crit_edge:		; preds = %no_exit.1.i.i
	br label %no_exit.2.i.i

no_exit.2.i.i:		; preds = %no_exit.2.i.i.no_exit.2.i.i_crit_edge, %loopentry.2.i.i.no_exit.2.i.i_crit_edge
	br bool false, label %no_exit.2.i.i.no_exit.2.i.i_crit_edge, label %no_exit.2.i.i.loopexit.2.i.i_crit_edge

no_exit.2.i.i.loopexit.2.i.i_crit_edge:		; preds = %no_exit.2.i.i
	br label %loopexit.2.i.i

no_exit.2.i.i.no_exit.2.i.i_crit_edge:		; preds = %no_exit.2.i.i
	br label %no_exit.2.i.i

loopexit.2.i.i:		; preds = %no_exit.2.i.i.loopexit.2.i.i_crit_edge, %loopentry.2.i.i.loopexit.2.i.i_crit_edge
	br bool false, label %loopexit.2.i.i.no_exit.1.i.i_crit_edge, label %loopexit.2.i.i.avoid_irq_share.exit.i_crit_edge

loopexit.2.i.i.avoid_irq_share.exit.i_crit_edge:		; preds = %loopexit.2.i.i
	br label %avoid_irq_share.exit.i

loopexit.2.i.i.no_exit.1.i.i_crit_edge:		; preds = %loopexit.2.i.i
	br label %no_exit.1.i.i

avoid_irq_share.exit.i:		; preds = %loopexit.2.i.i.avoid_irq_share.exit.i_crit_edge, %loopentry.1.i.i.avoid_irq_share.exit.i_crit_edge
	br label %endif.6.i

endif.6.i:		; preds = %avoid_irq_share.exit.i, %endif.3.i.endif.6.i_crit_edge
	br label %loopcont.0.i

loopcont.0.i:		; preds = %endif.6.i, %then.5.i, %then.1.i
	br bool false, label %loopcont.0.i.no_exit.0.i_crit_edge, label %loopcont.0.i.loopexit.0.i_crit_edge

loopcont.0.i.loopexit.0.i_crit_edge:		; preds = %loopcont.0.i
	br label %loopexit.0.i

loopcont.0.i.no_exit.0.i_crit_edge:		; preds = %loopcont.0.i
	br label %no_exit.0.i

loopexit.0.i:		; preds = %loopcont.0.i.loopexit.0.i_crit_edge, %loopentry.0.i.loopexit.0.i_crit_edge
	ret void

probe_serial_pnp.exit:		; No predecessors!
	unreachable

after_ret:		; No predecessors!
	ret void

return:		; No predecessors!
	unreachable
}
