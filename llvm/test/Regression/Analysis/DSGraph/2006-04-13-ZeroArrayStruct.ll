; RUN: llvm-as < %s | opt -analyze -datastructure

; ModuleID = 'bug3.bc'
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"

	%struct.Qdisc = type { 
int (%struct.sk_buff*, %struct.Qdisc*)*, 
%struct.sk_buff* (%struct.Qdisc*)*, 
uint, 
%struct.Qdisc_ops*, 
%struct.Qdisc*, 
uint, 
%typedef.atomic_t, 
%struct.sk_buff_head, 
%struct.net_device*, 
%struct.tc_stats, 
int (%struct.sk_buff*, %struct.Qdisc*)*, 
%struct.Qdisc*, 
[0 x sbyte] }

	%struct.Qdisc_class_ops = type { int (%struct.Qdisc*, uint, %struct.Qdisc*, %struct.Qdisc**)*, %struct.Qdisc* (%struct.Qdisc*, uint)*, uint (%struct.Qdisc*, uint)*, void (%struct.Qdisc*, uint)*, int (%struct.Qdisc*, uint, uint, %struct.rtattr**, uint*)*, int (%struct.Qdisc*, uint)*, void (%struct.Qdisc*, %struct.qdisc_walker*)*, %struct.tcf_proto** (%struct.Qdisc*, uint)*, uint (%struct.Qdisc*, uint, uint)*, void (%struct.Qdisc*, uint)*, int (%struct.Qdisc*, uint, %struct.sk_buff*, %struct.tcmsg*)* }
	%struct.Qdisc_ops = type { %struct.Qdisc_ops*, %struct.Qdisc_class_ops*, [16 x sbyte], int, int (%struct.sk_buff*, %struct.Qdisc*)*, %struct.sk_buff* (%struct.Qdisc*)*, int (%struct.sk_buff*, %struct.Qdisc*)*, uint (%struct.Qdisc*)*, int (%struct.Qdisc*, %struct.rtattr*)*, void (%struct.Qdisc*)*, void (%struct.Qdisc*)*, int (%struct.Qdisc*, %struct.rtattr*)*, int (%struct.Qdisc*, %struct.sk_buff*)* }
	%struct.ViceFid = type { uint, uint, uint }
	%struct.__wait_queue_head = type { %struct.icmp_filter, %struct.list_head }
	%struct.address_space = type { %struct.list_head, %struct.list_head, %struct.list_head, uint, %struct.address_space_operations*, %struct.inode*, %struct.vm_area_struct*, %struct.vm_area_struct*, %struct.icmp_filter, int }
	%struct.address_space_operations = type { int (%struct.page*)*, int (%struct.file*, %struct.page*)*, int (%struct.page*)*, int (%struct.file*, %struct.page*, uint, uint)*, int (%struct.file*, %struct.page*, uint, uint)*, int (%struct.address_space*, int)*, int (%struct.page*, uint)*, int (%struct.page*, int)*, int (int, %struct.inode*, %struct.kiobuf*, uint, int)*, int (int, %struct.file*, %struct.kiobuf*, uint, int)*, void (%struct.page*)* }
	%struct.affs_bm_info = type { uint, uint }
	%struct.block_device = type { %struct.list_head, %typedef.atomic_t, %struct.inode*, ushort, int, %struct.block_device_operations*, %struct.semaphore, %struct.list_head }
	%struct.block_device_operations = type opaque
	%struct.buffer_head = type { %struct.buffer_head*, uint, ushort, ushort, ushort, %typedef.atomic_t, ushort, uint, uint, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head*, %struct.buffer_head**, sbyte*, %struct.page*, void (%struct.buffer_head*, int)*, sbyte*, uint, %struct.__wait_queue_head, %struct.list_head }
	%struct.char_device = type { %struct.list_head, %typedef.atomic_t, ushort, %typedef.atomic_t, %struct.semaphore }
	%struct.completion = type { uint, %struct.__wait_queue_head }
	%struct.ctl_table = type { int, sbyte*, sbyte*, int, ushort, %struct.ctl_table*, int (%struct.ctl_table*, int, %struct.file*, sbyte*, uint*)*, int (%struct.ctl_table*, int*, int, sbyte*, uint*, sbyte*, uint, sbyte**)*, %struct.proc_dir_entry*, sbyte*, sbyte* }
	%struct.dentry = type { %typedef.atomic_t, uint, %struct.inode*, %struct.dentry*, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, int, %struct.qstr, uint, %struct.dentry_operations*, %struct.super_block*, uint, sbyte*, [16 x ubyte] }
	%struct.dentry_operations = type { int (%struct.dentry*, int)*, int (%struct.dentry*, %struct.qstr*)*, int (%struct.dentry*, %struct.qstr*, %struct.qstr*)*, int (%struct.dentry*)*, void (%struct.dentry*)*, void (%struct.dentry*, %struct.inode*)* }
	%struct.dev_mc_list = type { %struct.dev_mc_list*, [8 x ubyte], ubyte, int, int }
	%struct.dnotify_struct = type opaque
	%struct.dquot = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.__wait_queue_head, %struct.__wait_queue_head, int, int, %struct.super_block*, uint, ushort, long, short, short, uint, %struct.mem_dqblk }
	%struct.dquot_operations = type { void (%struct.inode*, int)*, void (%struct.inode*)*, int (%struct.inode*, ulong, int)*, int (%struct.inode*, uint)*, void (%struct.inode*, ulong)*, void (%struct.inode*, uint)*, int (%struct.inode*, %struct.iattr*)*, int (%struct.dquot*)* }
	%struct.dst_entry = type { %struct.dst_entry*, %typedef.atomic_t, int, %struct.net_device*, int, int, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, int, %struct.neighbour*, %struct.hh_cache*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)*, %struct.dst_ops*, [0 x sbyte] }
	%struct.dst_ops = type { ushort, ushort, uint, int ()*, %struct.dst_entry* (%struct.dst_entry*, uint)*, %struct.dst_entry* (%struct.dst_entry*, %struct.sk_buff*)*, void (%struct.dst_entry*)*, %struct.dst_entry* (%struct.dst_entry*)*, void (%struct.sk_buff*)*, int, %typedef.atomic_t, %struct.kmem_cache_s* }
	%struct.exec_domain = type opaque
	%struct.ext2_inode_info = type { [15 x uint], uint, uint, ubyte, ubyte, uint, uint, uint, uint, uint, uint, uint, uint, uint, int }
	%struct.ext3_inode_info = type { [15 x uint], uint, uint, uint, uint, uint, uint, uint, uint, uint, %struct.list_head, long, %struct.rw_semaphore }
	%struct.fasync_struct = type { int, int, %struct.fasync_struct*, %struct.file* }
	%struct.file = type { %struct.list_head, %struct.dentry*, %struct.vfsmount*, %struct.file_operations*, %typedef.atomic_t, uint, ushort, long, uint, uint, uint, uint, uint, %struct.fown_struct, uint, uint, int, uint, sbyte*, %struct.kiobuf*, int }
	%struct.file_lock = type { %struct.file_lock*, %struct.list_head, %struct.list_head, %struct.files_struct*, uint, %struct.__wait_queue_head, %struct.file*, ubyte, ubyte, long, long, void (%struct.file_lock*)*, void (%struct.file_lock*)*, void (%struct.file_lock*)*, %struct.fasync_struct*, uint, { %struct.nfs_lock_info } }
	%struct.file_operations = type { %struct.module*, long (%struct.file*, long, int)*, int (%struct.file*, sbyte*, uint, long*)*, int (%struct.file*, sbyte*, uint, long*)*, int (%struct.file*, sbyte*, int (sbyte*, sbyte*, int, long, uint, uint)*)*, uint (%struct.file*, %struct.poll_table_struct*)*, int (%struct.inode*, %struct.file*, uint, uint)*, int (%struct.file*, %struct.vm_area_struct*)*, int (%struct.inode*, %struct.file*)*, int (%struct.file*)*, int (%struct.inode*, %struct.file*)*, int (%struct.file*, %struct.dentry*, int)*, int (int, %struct.file*, int)*, int (%struct.file*, int, %struct.file_lock*)*, int (%struct.file*, %struct.iovec*, uint, long*)*, int (%struct.file*, %struct.iovec*, uint, long*)*, int (%struct.file*, %struct.page*, int, uint, long*, int)*, uint (%struct.file*, uint, uint, uint, uint)* }
	%struct.file_system_type = type { sbyte*, int, %struct.super_block* (%struct.super_block*, sbyte*, int)*, %struct.module*, %struct.file_system_type*, %struct.list_head }
	%struct.files_struct = type { %typedef.atomic_t, %typedef.rwlock_t, int, int, int, %struct.file**, %typedef.__kernel_fd_set*, %typedef.__kernel_fd_set*, %typedef.__kernel_fd_set, %typedef.__kernel_fd_set, [32 x %struct.file*] }
	%struct.fown_struct = type { int, uint, uint, int }
	%struct.fs_disk_quota = type { sbyte, sbyte, ushort, uint, ulong, ulong, ulong, ulong, ulong, ulong, int, int, ushort, ushort, int, ulong, ulong, ulong, int, ushort, short, [8 x sbyte] }
	%struct.fs_qfilestat = type { ulong, ulong, uint }
	%struct.fs_quota_stat = type { sbyte, ushort, sbyte, %struct.fs_qfilestat, %struct.fs_qfilestat, uint, int, int, int, ushort, ushort }
	%struct.fs_struct = type { %typedef.atomic_t, %typedef.rwlock_t, int, %struct.dentry*, %struct.dentry*, %struct.dentry*, %struct.vfsmount*, %struct.vfsmount*, %struct.vfsmount* }
	%struct.hh_cache = type { %struct.hh_cache*, %typedef.atomic_t, ushort, int, int (%struct.sk_buff*)*, %typedef.rwlock_t, [32 x uint] }
	%struct.i387_fxsave_struct = type { ushort, ushort, ushort, ushort, int, int, int, int, int, int, [32 x int], [32 x int], [56 x int] }
	%struct.iattr = type { uint, ushort, uint, uint, long, int, int, int, uint }
	%struct.icmp_filter = type { uint }
	%struct.if_dqblk = type { ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, uint }
	%struct.if_dqinfo = type { ulong, ulong, uint, uint }
	%struct.ifmap = type { uint, uint, ushort, ubyte, ubyte, ubyte }
	%struct.ifreq = type { { [16 x sbyte] }, { [2 x ulong] } }
	%struct.inode = type { %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, uint, %typedef.atomic_t, ushort, ushort, ushort, uint, uint, ushort, long, int, int, int, uint, uint, uint, uint, ushort, %struct.semaphore, %struct.rw_semaphore, %struct.semaphore, %struct.inode_operations*, %struct.file_operations*, %struct.super_block*, %struct.__wait_queue_head, %struct.file_lock*, %struct.address_space*, %struct.address_space, [2 x %struct.dquot*], %struct.list_head, %struct.pipe_inode_info*, %struct.block_device*, %struct.char_device*, uint, %struct.dnotify_struct*, uint, uint, ubyte, %typedef.atomic_t, uint, uint, { %struct.ext2_inode_info, %struct.ext3_inode_info, %struct.msdos_inode_info, %struct.iso_inode_info, %struct.nfs_inode_info, %struct.shmem_inode_info, %struct.proc_inode_info, %struct.socket, %struct.usbdev_inode_info, sbyte* } }
	%struct.inode_operations = type { int (%struct.inode*, %struct.dentry*, int)*, %struct.dentry* (%struct.inode*, %struct.dentry*)*, int (%struct.dentry*, %struct.inode*, %struct.dentry*)*, int (%struct.inode*, %struct.dentry*)*, int (%struct.inode*, %struct.dentry*, sbyte*)*, int (%struct.inode*, %struct.dentry*, int)*, int (%struct.inode*, %struct.dentry*)*, int (%struct.inode*, %struct.dentry*, int, int)*, int (%struct.inode*, %struct.dentry*, %struct.inode*, %struct.dentry*)*, int (%struct.dentry*, sbyte*, int)*, int (%struct.dentry*, %struct.nameidata*)*, void (%struct.inode*)*, int (%struct.inode*, int)*, int (%struct.dentry*)*, int (%struct.dentry*, %struct.iattr*)*, int (%struct.dentry*, %struct.iattr*)*, int (%struct.dentry*, sbyte*, sbyte*, uint, int)*, int (%struct.dentry*, sbyte*, sbyte*, uint)*, int (%struct.dentry*, sbyte*, uint)*, int (%struct.dentry*, sbyte*)* }
	%struct.iovec = type { sbyte*, uint }
	%struct.ip_options = type { uint, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, [0 x ubyte] }
	%struct.iso_inode_info = type { uint, ubyte, [3 x ubyte], uint, int }
	%struct.iw_handler_def = type opaque
	%struct.iw_statistics = type opaque
	%struct.k_sigaction = type { %struct.sigaction }
	%struct.kern_ipc_perm = type { int, uint, uint, uint, uint, ushort, uint }
	%struct.kiobuf = type opaque
	%struct.kmem_cache_s = type opaque
	%struct.linger = type { int, int }
	%struct.linux_binfmt = type { %struct.linux_binfmt*, %struct.module*, int (%struct.linux_binprm*, %struct.pt_regs*)*, int (%struct.file*)*, int (int, %struct.pt_regs*, %struct.file*)*, uint, int (%struct.linux_binprm*, sbyte*)* }
	%struct.linux_binprm = type { [128 x sbyte], [32 x %struct.page*], uint, int, %struct.file*, int, int, uint, uint, uint, int, int, sbyte*, uint, uint }
	%struct.list_head = type { %struct.list_head*, %struct.list_head* }
	%struct.llva_sigcontext = type { %typedef.llva_icontext_t, %typedef.llva_fp_state_t, uint, uint, uint, uint, [1 x uint], sbyte* }
	%struct.mem_dqblk = type { uint, uint, ulong, uint, uint, uint, int, int }
	%struct.mem_dqinfo = type { %struct.quota_format_type*, int, uint, uint, { %struct.ViceFid } }
	%struct.mm_struct = type { %struct.vm_area_struct*, %struct.rb_root_s, %struct.vm_area_struct*, %struct.icmp_filter*, %typedef.atomic_t, %typedef.atomic_t, int, %struct.rw_semaphore, %struct.icmp_filter, %struct.list_head, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, %struct.iovec }
	%struct.module = type { uint, %struct.module*, sbyte*, uint, %typedef.atomic_t, uint, uint, uint, %struct.module_symbol*, %struct.module_ref*, %struct.module_ref*, int ()*, void ()*, %struct.affs_bm_info*, %struct.affs_bm_info*, %struct.module_persist*, %struct.module_persist*, int ()*, int, sbyte*, sbyte*, sbyte*, sbyte*, sbyte* }
	%struct.module_persist = type opaque
	%struct.module_ref = type { %struct.module*, %struct.module*, %struct.module_ref* }
	%struct.module_symbol = type { uint, sbyte* }
	%struct.msdos_inode_info = type { uint, int, int, int, int, int, %struct.inode*, %struct.list_head }
	%struct.msghdr = type { sbyte*, int, %struct.iovec*, uint, sbyte*, uint, uint }
	%struct.nameidata = type { %struct.dentry*, %struct.vfsmount*, %struct.qstr, uint, int }
	%struct.namespace = type opaque
	%struct.nda_cacheinfo = type { uint, uint, uint, uint }
	%struct.neigh_ops = type { int, void (%struct.neighbour*)*, void (%struct.neighbour*, %struct.sk_buff*)*, void (%struct.neighbour*, %struct.sk_buff*)*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)*, int (%struct.sk_buff*)* }
	%struct.neigh_parms = type { %struct.neigh_parms*, int (%struct.neighbour*)*, %struct.neigh_table*, int, sbyte*, sbyte*, int, int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.neigh_table = type { %struct.neigh_table*, int, int, int, uint (sbyte*, %struct.net_device*)*, int (%struct.neighbour*)*, int (%struct.pneigh_entry*)*, void (%struct.pneigh_entry*)*, void (%struct.sk_buff*)*, sbyte*, %struct.neigh_parms, int, int, int, int, uint, %struct.timer_list, %struct.timer_list, %struct.sk_buff_head, int, %typedef.rwlock_t, uint, %struct.neigh_parms*, %struct.kmem_cache_s*, %struct.tasklet_struct, %struct.nda_cacheinfo, [32 x %struct.neighbour*], [16 x %struct.pneigh_entry*] }
	%struct.neighbour = type { %struct.neighbour*, %struct.neigh_table*, %struct.neigh_parms*, %struct.net_device*, uint, uint, uint, ubyte, ubyte, ubyte, ubyte, %typedef.atomic_t, %typedef.rwlock_t, [8 x ubyte], %struct.hh_cache*, %typedef.atomic_t, int (%struct.sk_buff*)*, %struct.sk_buff_head, %struct.timer_list, %struct.neigh_ops*, [0 x ubyte] }
	%struct.net_bridge_port = type opaque
	%struct.net_device = type { [16 x sbyte], uint, uint, uint, uint, uint, uint, ubyte, ubyte, uint, %struct.net_device*, int (%struct.net_device*)*, %struct.net_device*, int, int, %struct.net_device_stats* (%struct.net_device*)*, %struct.iw_statistics* (%struct.net_device*)*, %struct.iw_handler_def*, uint, uint, ushort, ushort, ushort, ushort, uint, ushort, ushort, sbyte*, %struct.net_device*, [8 x ubyte], [8 x ubyte], ubyte, %struct.dev_mc_list*, int, int, int, int, %struct.timer_list, sbyte*, sbyte*, sbyte*, sbyte*, sbyte*, %struct.list_head, int, int, %struct.Qdisc*, %struct.Qdisc*, %struct.Qdisc*, %struct.Qdisc*, uint, %struct.icmp_filter, int, %struct.icmp_filter, %typedef.atomic_t, int, int, void (%struct.net_device*)*, void (%struct.net_device*)*, int (%struct.net_device*)*, int (%struct.net_device*)*, int (%struct.sk_buff*, %struct.net_device*)*, int (%struct.net_device*, int*)*, int (%struct.sk_buff*, %struct.net_device*, ushort, sbyte*, sbyte*, uint)*, int (%struct.sk_buff*)*, void (%struct.net_device*)*, int (%struct.net_device*, sbyte*)*, int (%struct.net_device*, %struct.ifreq*, int)*, int (%struct.net_device*, %struct.ifmap*)*, int (%struct.neighbour*, %struct.hh_cache*)*, void (%struct.hh_cache*, %struct.net_device*, ubyte*)*, int (%struct.net_device*, int)*, void (%struct.net_device*)*, void (%struct.net_device*, %struct.vlan_group*)*, void (%struct.net_device*, ushort)*, void (%struct.net_device*, ushort)*, int (%struct.sk_buff*, ubyte*)*, int (%struct.net_device*, %struct.neigh_parms*)*, int (%struct.net_device*, %struct.dst_entry*)*, %struct.module*, %struct.net_bridge_port* }
	%struct.net_device_stats = type { uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint }
	%struct.nf_conntrack = type { %typedef.atomic_t, void (%struct.nf_conntrack*)* }
	%struct.nf_ct_info = type { %struct.nf_conntrack* }
	%struct.nfs_fh = type { ushort, [64 x ubyte] }
	%struct.nfs_inode_info = type { ulong, %struct.nfs_fh, ushort, uint, ulong, ulong, ulong, uint, uint, uint, [2 x uint], %struct.list_head, %struct.list_head, %struct.list_head, %struct.list_head, uint, uint, uint, uint, %struct.rpc_cred* }
	%struct.nfs_lock_info = type { uint, uint, %struct.nlm_host* }
	%struct.nlm_host = type opaque
	%struct.notifier_block = type { int (%struct.notifier_block*, uint, sbyte*)*, %struct.notifier_block*, int }
	%struct.open_request = type { %struct.open_request*, uint, uint, ushort, ushort, ubyte, ubyte, ushort, uint, uint, uint, uint, %struct.or_calltable*, %struct.sock*, { %struct.tcp_v4_open_req } }
	%struct.or_calltable = type { int, int (%struct.sock*, %struct.open_request*, %struct.dst_entry*)*, void (%struct.sk_buff*, %struct.open_request*)*, void (%struct.open_request*)*, void (%struct.sk_buff*)* }
	%struct.page = type { %struct.list_head, %struct.address_space*, uint, %struct.page*, %typedef.atomic_t, uint, %struct.list_head, %struct.page**, %struct.buffer_head* }
	%struct.pipe_inode_info = type { %struct.__wait_queue_head, sbyte*, uint, uint, uint, uint, uint, uint, uint, uint }
	%struct.pneigh_entry = type { %struct.pneigh_entry*, %struct.net_device*, [0 x ubyte] }
	%struct.poll_table_page = type opaque
	%struct.poll_table_struct = type { int, %struct.poll_table_page* }
	%struct.proc_dir_entry = type { ushort, ushort, sbyte*, ushort, ushort, uint, uint, uint, %struct.inode_operations*, %struct.file_operations*, int (sbyte*, sbyte**, int, int)*, %struct.module*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, %struct.proc_dir_entry*, sbyte*, int (sbyte*, sbyte**, int, int, int*, sbyte*)*, int (%struct.file*, sbyte*, uint, sbyte*)*, %typedef.atomic_t, int, ushort }
	%struct.proc_inode_info = type { %struct.task_struct*, int, { int (%struct.task_struct*, sbyte*)* }, %struct.file* }
	%struct.proto = type { void (%struct.sock*, int)*, int (%struct.sock*, %struct.sockaddr*, int)*, int (%struct.sock*, int)*, %struct.sock* (%struct.sock*, int, int*)*, int (%struct.sock*, int, uint)*, int (%struct.sock*)*, int (%struct.sock*)*, void (%struct.sock*, int)*, int (%struct.sock*, int, int, sbyte*, int)*, int (%struct.sock*, int, int, sbyte*, int*)*, int (%struct.sock*, %struct.msghdr*, int)*, int (%struct.sock*, %struct.msghdr*, int, int, int, int*)*, int (%struct.sock*, %struct.sockaddr*, int)*, int (%struct.sock*, %struct.sk_buff*)*, void (%struct.sock*)*, void (%struct.sock*)*, int (%struct.sock*, ushort)*, [32 x sbyte], [32 x { int, [28 x ubyte] }] }
	%struct.proto_ops = type { int, int (%struct.socket*)*, int (%struct.socket*, %struct.sockaddr*, int)*, int (%struct.socket*, %struct.sockaddr*, int, int)*, int (%struct.socket*, %struct.socket*)*, int (%struct.socket*, %struct.socket*, int)*, int (%struct.socket*, %struct.sockaddr*, int*, int)*, uint (%struct.file*, %struct.socket*, %struct.poll_table_struct*)*, int (%struct.socket*, uint, uint)*, int (%struct.socket*, int)*, int (%struct.socket*, int)*, int (%struct.socket*, int, int, sbyte*, int)*, int (%struct.socket*, int, int, sbyte*, int*)*, int (%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)*, int (%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)*, int (%struct.file*, %struct.socket*, %struct.vm_area_struct*)*, int (%struct.socket*, %struct.page*, int, uint, int)* }
	%struct.pt_regs = type { int, int, int, int, int, int, int, int, int, int, int, int, int, int, int }
	%struct.qdisc_walker = type { int, int, int, int (%struct.Qdisc*, uint, %struct.qdisc_walker*)* }
	%struct.qstr = type { ubyte*, uint, uint }
	%struct.quota_format_ops = type { int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.dquot*)*, int (%struct.dquot*)* }
	%struct.quota_format_type = type opaque
	%struct.quota_info = type { uint, %struct.semaphore, %struct.semaphore, [2 x %struct.file*], [2 x %struct.mem_dqinfo], [2 x %struct.quota_format_ops*] }
	%struct.quotactl_ops = type { int (%struct.super_block*, int, int, sbyte*)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int)*, int (%struct.super_block*, int, %struct.if_dqinfo*)*, int (%struct.super_block*, int, %struct.if_dqinfo*)*, int (%struct.super_block*, int, uint, %struct.if_dqblk*)*, int (%struct.super_block*, int, uint, %struct.if_dqblk*)*, int (%struct.super_block*, %struct.fs_quota_stat*)*, int (%struct.super_block*, uint, int)*, int (%struct.super_block*, int, uint, %struct.fs_disk_quota*)*, int (%struct.super_block*, int, uint, %struct.fs_disk_quota*)* }
	%struct.rb_node_s = type { %struct.rb_node_s*, int, %struct.rb_node_s*, %struct.rb_node_s* }
	%struct.rb_root_s = type { %struct.rb_node_s* }
	%struct.revectored_struct = type { [8 x uint] }
	%struct.rpc_cred = type opaque
	%struct.rtattr = type { ushort, ushort }
	%struct.rw_semaphore = type { int, %struct.icmp_filter, %struct.list_head }
	%struct.scm_cookie = type { %struct.ViceFid, %struct.scm_fp_list*, uint }
	%struct.scm_fp_list = type { int, [255 x %struct.file*] }
	%struct.sem_array = type { %struct.kern_ipc_perm, int, int, %struct.linger*, %struct.sem_queue*, %struct.sem_queue**, %struct.sem_undo*, uint }
	%struct.sem_queue = type { %struct.sem_queue*, %struct.sem_queue**, %struct.task_struct*, %struct.sem_undo*, int, int, %struct.sem_array*, int, %struct.sembuf*, int, int }
	%struct.sem_undo = type { %struct.sem_undo*, %struct.sem_undo*, int, short* }
	%struct.semaphore = type { %typedef.atomic_t, int, %struct.__wait_queue_head }
	%struct.sembuf = type { ushort, short, short }
	%struct.seq_file = type opaque
	%struct.shmem_inode_info = type { %struct.icmp_filter, uint, [16 x %struct.icmp_filter], sbyte**, uint, uint, %struct.list_head, %struct.inode* }
	%struct.sigaction = type { void (int)*, uint, void ()*, %typedef.sigset_t }
	%struct.siginfo = type { int, int, int, { [29 x int] } }
	%struct.signal_struct = type { %typedef.atomic_t, [64 x %struct.k_sigaction], %struct.icmp_filter }
	%struct.sigpending = type { %struct.sigqueue*, %struct.sigqueue**, %typedef.sigset_t }
	%struct.sigqueue = type { %struct.sigqueue*, %struct.siginfo }
	%struct.sk_buff = type { %struct.sk_buff*, %struct.sk_buff*, %struct.sk_buff_head*, %struct.sock*, %struct.linger, %struct.net_device*, %struct.net_device*, { ubyte* }, { ubyte* }, { ubyte* }, %struct.dst_entry*, [48 x sbyte], uint, uint, uint, ubyte, ubyte, ubyte, ubyte, uint, %typedef.atomic_t, ushort, ushort, uint, ubyte*, ubyte*, ubyte*, ubyte*, void (%struct.sk_buff*)*, uint, uint, %struct.nf_ct_info*, uint }
	%struct.sk_buff_head = type { %struct.sk_buff*, %struct.sk_buff*, uint, %struct.icmp_filter }
	%struct.sock = type { uint, uint, ushort, ushort, int, %struct.sock*, %struct.sock**, %struct.sock*, %struct.sock**, ubyte, ubyte, ushort, ushort, ubyte, ubyte, %typedef.atomic_t, %typedef.socket_lock_t, int, %struct.__wait_queue_head*, %struct.dst_entry*, %typedef.rwlock_t, %typedef.atomic_t, %struct.sk_buff_head, %typedef.atomic_t, %struct.sk_buff_head, %typedef.atomic_t, int, int, uint, uint, int, %struct.sock*, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, sbyte, ubyte, ubyte, ubyte, ubyte, int, int, uint, int, %struct.sock*, { %struct.sk_buff*, %struct.sk_buff* }, %typedef.rwlock_t, %struct.sk_buff_head, %struct.proto*, { %struct.tcp_opt }, int, int, ushort, ushort, uint, ushort, ubyte, ubyte, %struct.ViceFid, int, int, int, { %struct.unix_opt }, %struct.timer_list, %struct.linger, %struct.socket*, sbyte*, void (%struct.sock*)*, void (%struct.sock*, int)*, void (%struct.sock*)*, void (%struct.sock*)*, int (%struct.sock*, %struct.sk_buff*)*, void (%struct.sock*)* }
	%struct.sockaddr = type { ushort, [14 x sbyte] }
	%struct.sockaddr_un = type { ushort, [108 x sbyte] }
	%struct.socket = type { uint, uint, %struct.proto_ops*, %struct.inode*, %struct.fasync_struct*, %struct.file*, %struct.sock*, %struct.__wait_queue_head, short, ubyte }
	%struct.statfs = type { int, int, int, int, int, int, int, %typedef.__kernel_fsid_t, int, [6 x int] }
	%struct.super_block = type { %struct.list_head, ushort, uint, ubyte, ubyte, ulong, %struct.file_system_type*, %struct.super_operations*, %struct.dquot_operations*, %struct.quotactl_ops*, uint, uint, %struct.dentry*, %struct.rw_semaphore, %struct.semaphore, int, %typedef.atomic_t, %struct.list_head, %struct.list_head, %struct.list_head, %struct.block_device*, %struct.list_head, %struct.quota_info, { [115 x uint] }, %struct.semaphore, %struct.semaphore }
	%struct.super_operations = type { %struct.inode* (%struct.super_block*)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.inode*, sbyte*)*, void (%struct.inode*)*, void (%struct.inode*, int)*, void (%struct.inode*)*, void (%struct.inode*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, int (%struct.super_block*)*, void (%struct.super_block*)*, void (%struct.super_block*)*, int (%struct.super_block*, %struct.statfs*)*, int (%struct.super_block*, int*, sbyte*)*, void (%struct.inode*)*, void (%struct.super_block*)*, %struct.dentry* (%struct.super_block*, uint*, int, int, int)*, int (%struct.dentry*, uint*, int*, int)*, int (%struct.seq_file*, %struct.vfsmount*)* }
	%struct.task_struct = type { int, uint, int, %struct.icmp_filter, %struct.exec_domain*, int, uint, int, int, int, uint, %struct.mm_struct*, int, uint, uint, %struct.list_head, uint, %struct.task_struct*, %struct.task_struct*, %struct.mm_struct*, %struct.list_head, uint, uint, %struct.linux_binfmt*, int, int, int, uint, int, int, int, int, int, int, int, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.task_struct*, %struct.list_head, %struct.task_struct*, %struct.task_struct**, %struct.__wait_queue_head, %struct.completion*, uint, uint, uint, uint, uint, uint, uint, %struct.timer_list, %struct.tms, uint, [32 x int], [32 x int], uint, uint, uint, uint, uint, uint, int, uint, uint, uint, uint, uint, uint, uint, uint, int, [32 x uint], uint, uint, uint, int, %struct.user_struct*, [11 x %struct.affs_bm_info], ushort, [16 x sbyte], int, int, %struct.tty_struct*, uint, %struct.sem_undo*, %struct.sem_queue*, %struct.thread_struct, %struct.fs_struct*, %struct.files_struct*, %struct.namespace*, %struct.icmp_filter, %struct.signal_struct*, %typedef.sigset_t, %struct.sigpending, uint, uint, int (sbyte*)*, sbyte*, %typedef.sigset_t*, uint, uint, %struct.icmp_filter, sbyte*, %struct.llva_sigcontext*, uint, %struct.task_struct*, uint, %typedef.llva_icontext_t, %typedef.llva_fp_state_t, uint*, int, sbyte* }
	%struct.tasklet_struct = type { %struct.tasklet_struct*, uint, %typedef.atomic_t, void (uint)*, uint }
	%struct.tc_stats = type { ulong, uint, uint, uint, uint, uint, uint, uint, %struct.icmp_filter* }
	%struct.tcf_proto = type { %struct.tcf_proto*, sbyte*, int (%struct.sk_buff*, %struct.tcf_proto*, %struct.affs_bm_info*)*, uint, uint, uint, %struct.Qdisc*, sbyte*, %struct.tcf_proto_ops* }
	%struct.tcf_proto_ops = type { %struct.tcf_proto_ops*, [16 x sbyte], int (%struct.sk_buff*, %struct.tcf_proto*, %struct.affs_bm_info*)*, int (%struct.tcf_proto*)*, void (%struct.tcf_proto*)*, uint (%struct.tcf_proto*, uint)*, void (%struct.tcf_proto*, uint)*, int (%struct.tcf_proto*, uint, uint, %struct.rtattr**, uint*)*, int (%struct.tcf_proto*, uint)*, void (%struct.tcf_proto*, %struct.tcf_walker*)*, int (%struct.tcf_proto*, uint, %struct.sk_buff*, %struct.tcmsg*)* }
	%struct.tcf_walker = type { int, int, int, int (%struct.tcf_proto*, uint, %struct.tcf_walker*)* }
	%struct.tcmsg = type { ubyte, ubyte, ushort, int, uint, uint, uint }
	%struct.tcp_bind_bucket = type { ushort, short, %struct.tcp_bind_bucket*, %struct.sock*, %struct.tcp_bind_bucket** }
	%struct.tcp_bind_hashbucket = type { %struct.icmp_filter, %struct.tcp_bind_bucket* }
	%struct.tcp_ehash_bucket = type { %typedef.rwlock_t, %struct.sock* }
	%struct.tcp_func = type { int (%struct.sk_buff*)*, void (%struct.sock*, %struct.tcphdr*, int, %struct.sk_buff*)*, int (%struct.sock*)*, int (%struct.sock*, %struct.sk_buff*)*, %struct.sock* (%struct.sock*, %struct.sk_buff*, %struct.open_request*, %struct.dst_entry*)*, int (%struct.sock*)*, ushort, int (%struct.sock*, int, int, sbyte*, int)*, int (%struct.sock*, int, int, sbyte*, int*)*, void (%struct.sock*, %struct.sockaddr*)*, int }
	%struct.tcp_hashinfo = type { %struct.tcp_ehash_bucket*, %struct.tcp_bind_hashbucket*, int, int, [32 x %struct.sock*], %typedef.rwlock_t, %typedef.atomic_t, %struct.__wait_queue_head, %struct.icmp_filter }
	%struct.tcp_listen_opt = type { ubyte, int, int, int, uint, [512 x %struct.open_request*] }
	%struct.tcp_opt = type { int, uint, uint, uint, uint, uint, uint, uint, { ubyte, ubyte, ubyte, ubyte, uint, uint, uint, ushort, ushort }, { %struct.sk_buff_head, %struct.task_struct*, %struct.iovec*, int, int }, uint, uint, uint, uint, ushort, ushort, ushort, ubyte, ubyte, ubyte, ubyte, ubyte, ubyte, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, ushort, ushort, uint, uint, uint, %struct.timer_list, %struct.timer_list, %struct.sk_buff_head, %struct.tcp_func*, %struct.sk_buff*, %struct.page*, uint, uint, uint, uint, uint, uint, sbyte, sbyte, sbyte, sbyte, ubyte, ubyte, ubyte, ubyte, uint, uint, uint, int, ushort, ubyte, ubyte, [1 x %struct.affs_bm_info], [4 x %struct.affs_bm_info], uint, uint, ubyte, ubyte, ushort, ubyte, ubyte, ushort, uint, uint, uint, uint, uint, uint, int, uint, ushort, ubyte, ubyte, uint, %typedef.rwlock_t, %struct.tcp_listen_opt*, %struct.open_request*, %struct.open_request*, int, uint, uint, int, int, uint, uint }
	%struct.tcp_v4_open_req = type { uint, uint, %struct.ip_options* }
	%struct.tcphdr = type { ushort, ushort, uint, uint, ushort, ushort, ushort, ushort }
	%struct.termios = type { uint, uint, uint, uint, ubyte, [19 x ubyte] }
	%struct.thread_struct = type { uint, uint, uint, uint, uint, [8 x uint], uint, uint, uint, %union.i387_union, %struct.vm86_struct*, uint, uint, uint, uint, int, [33 x uint] }
	%struct.timer_list = type { %struct.list_head, uint, uint, void (uint)* }
	%struct.tms = type { int, int, int, int }
	%struct.tq_struct = type { %struct.list_head, uint, void (sbyte*)*, sbyte* }
	%struct.tty_driver = type { int, sbyte*, sbyte*, int, short, short, short, short, short, %struct.termios, int, int*, %struct.proc_dir_entry*, %struct.tty_driver*, %struct.tty_struct**, %struct.termios**, %struct.termios**, sbyte*, int (%struct.tty_struct*, %struct.file*)*, void (%struct.tty_struct*, %struct.file*)*, int (%struct.tty_struct*, int, ubyte*, int)*, void (%struct.tty_struct*, ubyte)*, void (%struct.tty_struct*)*, int (%struct.tty_struct*)*, int (%struct.tty_struct*)*, int (%struct.tty_struct*, %struct.file*, uint, uint)*, void (%struct.tty_struct*, %struct.termios*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, int)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*, int)*, void (%struct.tty_struct*, sbyte)*, int (sbyte*, sbyte**, int, int, int*, sbyte*)*, int (%struct.file*, sbyte*, uint, sbyte*)*, %struct.tty_driver*, %struct.tty_driver* }
	%struct.tty_flip_buffer = type { %struct.tq_struct, %struct.semaphore, sbyte*, ubyte*, int, int, [1024 x ubyte], [1024 x sbyte], [4 x ubyte] }
	%struct.tty_ldisc = type { int, sbyte*, int, int, int (%struct.tty_struct*)*, void (%struct.tty_struct*)*, void (%struct.tty_struct*)*, int (%struct.tty_struct*)*, int (%struct.tty_struct*, %struct.file*, ubyte*, uint)*, int (%struct.tty_struct*, %struct.file*, ubyte*, uint)*, int (%struct.tty_struct*, %struct.file*, uint, uint)*, void (%struct.tty_struct*, %struct.termios*)*, uint (%struct.tty_struct*, %struct.file*, %struct.poll_table_struct*)*, void (%struct.tty_struct*, ubyte*, sbyte*, int)*, int (%struct.tty_struct*)*, void (%struct.tty_struct*)* }
	%struct.tty_struct = type { int, %struct.tty_driver, %struct.tty_ldisc, %struct.termios*, %struct.termios*, int, int, ushort, uint, int, %struct.udphdr, ubyte, ubyte, %struct.tty_struct*, %struct.fasync_struct*, %struct.tty_flip_buffer, int, int, %struct.__wait_queue_head, %struct.__wait_queue_head, %struct.tq_struct, sbyte*, sbyte*, %struct.list_head, uint, ubyte, ushort, uint, int, [8 x uint], sbyte*, int, int, int, [128 x uint], int, uint, uint, %struct.semaphore, %struct.semaphore, %struct.icmp_filter, %struct.tq_struct }
	%struct.udphdr = type { ushort, ushort, ushort, ushort }
	%struct.unix_address = type { %typedef.atomic_t, int, uint, [0 x %struct.sockaddr_un] }
	%struct.unix_opt = type { %struct.unix_address*, %struct.dentry*, %struct.vfsmount*, %struct.semaphore, %struct.sock*, %struct.sock**, %struct.sock*, %typedef.atomic_t, %typedef.rwlock_t, %struct.__wait_queue_head }
	%struct.usb_bus = type opaque
	%struct.usbdev_inode_info = type { %struct.list_head, %struct.list_head, { %struct.usb_bus* } }
	%struct.user_struct = type { %typedef.atomic_t, %typedef.atomic_t, %typedef.atomic_t, %struct.user_struct*, %struct.user_struct**, uint }
	%struct.vfsmount = type { %struct.list_head, %struct.vfsmount*, %struct.dentry*, %struct.dentry*, %struct.super_block*, %struct.list_head, %struct.list_head, %typedef.atomic_t, int, sbyte*, %struct.list_head }
	%struct.vlan_group = type opaque
	%struct.vm86_regs = type { int, int, int, int, int, int, int, int, int, int, int, int, int, ushort, ushort, int, int, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort, ushort }
	%struct.vm86_struct = type { %struct.vm86_regs, uint, uint, uint, %struct.revectored_struct, %struct.revectored_struct }
	%struct.vm_area_struct = type { %struct.mm_struct*, uint, uint, %struct.vm_area_struct*, %struct.icmp_filter, uint, %struct.rb_node_s, %struct.vm_area_struct*, %struct.vm_area_struct**, %struct.vm_operations_struct*, uint, %struct.file*, uint, sbyte* }
	%struct.vm_operations_struct = type { void (%struct.vm_area_struct*)*, void (%struct.vm_area_struct*)*, %struct.page* (%struct.vm_area_struct*, uint, int)* }
	%typedef.__kernel_fd_set = type { [32 x int] }
	%typedef.__kernel_fsid_t = type { [2 x int] }
	%typedef.atomic_t = type { int }
	%typedef.llva_fp_state_t = type { [7 x uint], [20 x uint] }
	%typedef.llva_icontext_t = type { uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint, uint*, uint }
	%typedef.rwlock_t = type { %struct.icmp_filter, %struct.icmp_filter, uint }
	%typedef.sigset_t = type { [2 x uint] }
	%typedef.socket_lock_t = type { %struct.icmp_filter, uint, %struct.__wait_queue_head }
	%union.i387_union = type { %struct.i387_fxsave_struct }
%noqueue_qdisc_ops = global %struct.Qdisc_ops {
    %struct.Qdisc_ops* null, 
    %struct.Qdisc_class_ops* null, 
    [16 x sbyte] c"noqueue\00\00\00\00\00\00\00\00\00", 
    int 0, 
    int (%struct.sk_buff*, %struct.Qdisc*)* %noop_enqueue, 
    %struct.sk_buff* (%struct.Qdisc*)* %noop_dequeue, 
    int (%struct.sk_buff*, %struct.Qdisc*)* %noop_requeue, 
    uint (%struct.Qdisc*)* null, 
    int (%struct.Qdisc*, %struct.rtattr*)* null, 
    void (%struct.Qdisc*)* null, 
    void (%struct.Qdisc*)* null, 
    int (%struct.Qdisc*, %struct.rtattr*)* null, 
    int (%struct.Qdisc*, %struct.sk_buff*)* null }		; <%struct.Qdisc_ops*> [#uses=1]
%noqueue_qdisc = global %struct.Qdisc {
    int (%struct.sk_buff*, %struct.Qdisc*)* null, 
    %struct.sk_buff* (%struct.Qdisc*)* %noop_dequeue, 
    uint 1, 
    %struct.Qdisc_ops* %noqueue_qdisc_ops, 
    %struct.Qdisc* null, 
    uint 0, 
    %typedef.atomic_t zeroinitializer, 
    %struct.sk_buff_head zeroinitializer, 
    %struct.net_device* null, 
    %struct.tc_stats zeroinitializer, 
    int (%struct.sk_buff*, %struct.Qdisc*)* null, 
    %struct.Qdisc* null, 
    [0 x sbyte] zeroinitializer }		; <%struct.Qdisc*> [#uses=0]
%tcp_hashinfo = global %struct.tcp_hashinfo {
    %struct.tcp_ehash_bucket* null, 
    %struct.tcp_bind_hashbucket* null, 
    int 0, 
    int 0, 
    [32 x %struct.sock*] zeroinitializer, 
    %typedef.rwlock_t {
        %struct.icmp_filter { uint 1 }, 
        %struct.icmp_filter { uint 1 }, 
        uint 0 }, 
    %typedef.atomic_t zeroinitializer, 
    %struct.__wait_queue_head { %struct.icmp_filter { uint 1 }, %struct.list_head { %struct.list_head* getelementptr (%struct.tcp_hashinfo* %tcp_hashinfo, int 0, uint 7, uint 1), %struct.list_head* getelementptr (%struct.tcp_hashinfo* %tcp_hashinfo, int 0, uint 7, uint 1) } }, 
    %struct.icmp_filter { uint 1 } }		; <%struct.tcp_hashinfo*> [#uses=1]
%arp_tbl = global %struct.neigh_table {
    %struct.neigh_table* null, 
    int 2, 
    int 112, 
    int 4, 
    uint (sbyte*, %struct.net_device*)* %arp_hash, 
    int (%struct.neighbour*)* %arp_constructor, 
    int (%struct.pneigh_entry*)* null, 
    void (%struct.pneigh_entry*)* null, 
    void (%struct.sk_buff*)* %parp_redo, 
    sbyte* getelementptr ([10 x sbyte]* %.str_1, int 0, int 0), 
    %struct.neigh_parms {
        %struct.neigh_parms* null, 
        int (%struct.neighbour*)* null, 
        %struct.neigh_table* %arp_tbl, 
        int 0, 
        sbyte* null, 
        sbyte* null, 
        int 3000, 
        int 100, 
        int 6000, 
        int 3000, 
        int 500, 
        int 3, 
        int 3, 
        int 0, 
        int 3, 
        int 100, 
        int 80, 
        int 64, 
        int 100 }, 
    int 3000, 
    int 128, 
    int 512, 
    int 1024, 
    uint 0, 
    %struct.timer_list zeroinitializer, 
    %struct.timer_list zeroinitializer, 
    %struct.sk_buff_head zeroinitializer, 
    int 0, 
    %typedef.rwlock_t zeroinitializer, 
    uint 0, 
    %struct.neigh_parms* null, 
    %struct.kmem_cache_s* null, 
    %struct.tasklet_struct zeroinitializer, 
    %struct.nda_cacheinfo zeroinitializer, 
    [32 x %struct.neighbour*] zeroinitializer, 
    [16 x %struct.pneigh_entry*] zeroinitializer }		; <%struct.neigh_table*> [#uses=1]
%.str_1 = internal global [10 x sbyte] c"arp_cache\00"		; <[10 x sbyte]*> [#uses=1]

implementation   ; Functions:

declare int %sock_no_connect(%struct.socket*, %struct.sockaddr*, int, int)

declare int %sock_no_socketpair(%struct.socket*, %struct.socket*)

declare int %sock_no_accept(%struct.socket*, %struct.socket*, int)

declare int %sock_no_ioctl(%struct.socket*, uint, uint)

declare int %sock_no_listen(%struct.socket*, int)

declare int %sock_no_shutdown(%struct.socket*, int)

declare int %sock_no_setsockopt(%struct.socket*, int, int, sbyte*, int)

declare int %sock_no_getsockopt(%struct.socket*, int, int, sbyte*, int*)

declare int %sock_no_mmap(%struct.file*, %struct.socket*, %struct.vm_area_struct*)

declare int %sock_no_sendpage(%struct.socket*, %struct.page*, int, uint, int)

declare uint %datagram_poll(%struct.file*, %struct.socket*, %struct.poll_table_struct*)

declare int %proc_dointvec(%struct.ctl_table*, int, %struct.file*, sbyte*, uint*)

declare int %proc_dointvec_jiffies(%struct.ctl_table*, int, %struct.file*, sbyte*, uint*)

declare int %dev_queue_xmit(%struct.sk_buff*)

declare int %dst_dev_event(%struct.notifier_block*, uint, sbyte*)

declare int %neigh_compat_output(%struct.sk_buff*)

declare int %rtnetlink_event(%struct.notifier_block*, uint, sbyte*)

declare int %noop_enqueue(%struct.sk_buff*, %struct.Qdisc*)

declare %struct.sk_buff* %noop_dequeue(%struct.Qdisc*)

declare int %noop_requeue(%struct.sk_buff*, %struct.Qdisc*)

declare int %netlink_create(%struct.socket*, int)

declare int %netlink_release(%struct.socket*)

declare int %netlink_bind(%struct.socket*, %struct.sockaddr*, int)

declare int %netlink_connect(%struct.socket*, %struct.sockaddr*, int, int)

declare int %netlink_getname(%struct.socket*, %struct.sockaddr*, int*, int)

declare int %netlink_sendmsg(%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)

declare int %netlink_recvmsg(%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)

declare int %rt_garbage_collect()

declare %struct.dst_entry* %ipv4_dst_check(%struct.dst_entry*, uint)

declare %struct.dst_entry* %ipv4_dst_reroute(%struct.dst_entry*, %struct.sk_buff*)

declare void %ipv4_dst_destroy(%struct.dst_entry*)

declare %struct.dst_entry* %ipv4_negative_advice(%struct.dst_entry*)

declare void %ipv4_link_failure(%struct.sk_buff*)

declare void %rt_check_expire__thr(uint)

declare void %rt_run_flush__thr(uint)

declare int %ipv4_sysctl_rtcache_flush(%struct.ctl_table*, int, %struct.file*, sbyte*, uint*)

declare int %ipv4_sysctl_rtcache_flush_strategy(%struct.ctl_table*, int*, int, sbyte*, uint*, sbyte*, uint, sbyte**)

declare int %sysctl_jiffies(%struct.ctl_table*, int*, int, sbyte*, uint*, sbyte*, uint, sbyte**)

declare int %tcp_v4_rcv(%struct.sk_buff*)

declare void %tcp_v4_err(%struct.sk_buff*, uint)

declare int %udp_rcv(%struct.sk_buff*)

declare void %udp_err(%struct.sk_buff*, uint)

declare int %icmp_rcv(%struct.sk_buff*)

declare int %ip_queue_xmit(%struct.sk_buff*)

declare int %ip_setsockopt(%struct.sock*, int, int, sbyte*, int)

declare int %ip_getsockopt(%struct.sock*, int, int, sbyte*, int*)

declare uint %tcp_poll(%struct.file*, %struct.socket*, %struct.poll_table_struct*)

declare int %tcp_ioctl(%struct.sock*, int, uint)

declare int %tcp_disconnect(%struct.sock*, int)

declare int %tcp_sendpage(%struct.socket*, %struct.page*, int, uint, int)

declare int %tcp_sendmsg(%struct.sock*, %struct.msghdr*, int)

declare int %tcp_recvmsg(%struct.sock*, %struct.msghdr*, int, int, int, int*)

declare void %tcp_shutdown(%struct.sock*, int)

declare void %tcp_close(%struct.sock*, int)

declare %struct.sock* %tcp_accept(%struct.sock*, int, int*)

declare int %tcp_setsockopt(%struct.sock*, int, int, sbyte*, int)

declare int %tcp_getsockopt(%struct.sock*, int, int, sbyte*, int*)

declare int %tcp_v4_get_port(%struct.sock*, ushort)

declare void %tcp_v4_hash(%struct.sock*)

declare void %tcp_unhash(%struct.sock*)

declare int %tcp_v4_connect(%struct.sock*, %struct.sockaddr*, int)

declare void %tcp_v4_send_check(%struct.sock*, %struct.tcphdr*, int, %struct.sk_buff*)

declare void %tcp_v4_send_reset(%struct.sk_buff*)

declare void %tcp_v4_or_send_ack(%struct.sk_buff*, %struct.open_request*)

declare int %tcp_v4_send_synack(%struct.sock*, %struct.open_request*, %struct.dst_entry*)

declare void %tcp_v4_or_free(%struct.open_request*)

declare int %tcp_v4_conn_request(%struct.sock*, %struct.sk_buff*)

declare %struct.sock* %tcp_v4_syn_recv_sock(%struct.sock*, %struct.sk_buff*, %struct.open_request*, %struct.dst_entry*)

declare int %tcp_v4_do_rcv(%struct.sock*, %struct.sk_buff*)

declare int %tcp_v4_rebuild_header(%struct.sock*)

declare void %v4_addr2sockaddr(%struct.sock*, %struct.sockaddr*)

declare int %tcp_v4_remember_stamp(%struct.sock*)

declare int %tcp_v4_init_sock(%struct.sock*)

declare int %tcp_v4_destroy_sock(%struct.sock*)

declare void %tcp_twkill__thr(uint)

declare void %tcp_twcal_tick__thr(uint)

declare void %raw_v4_hash(%struct.sock*)

declare void %raw_v4_unhash(%struct.sock*)

declare int %raw_rcv_skb(%struct.sock*, %struct.sk_buff*)

declare int %raw_sendmsg(%struct.sock*, %struct.msghdr*, int)

declare void %raw_close(%struct.sock*, int)

declare int %raw_bind(%struct.sock*, %struct.sockaddr*, int)

declare int %raw_recvmsg(%struct.sock*, %struct.msghdr*, int, int, int, int*)

declare int %raw_init(%struct.sock*)

declare int %raw_setsockopt(%struct.sock*, int, int, sbyte*, int)

declare int %raw_getsockopt(%struct.sock*, int, int, sbyte*, int*)

declare int %raw_ioctl(%struct.sock*, int, uint)

declare int %udp_connect(%struct.sock*, %struct.sockaddr*, int)

declare int %udp_disconnect(%struct.sock*, int)

declare int %udp_v4_get_port(%struct.sock*, ushort)

declare void %udp_v4_hash(%struct.sock*)

declare void %udp_v4_unhash(%struct.sock*)

declare int %udp_sendmsg(%struct.sock*, %struct.msghdr*, int)

declare int %udp_ioctl(%struct.sock*, int, uint)

declare int %udp_recvmsg(%struct.sock*, %struct.msghdr*, int, int, int, int*)

declare void %udp_close(%struct.sock*, int)

declare int %udp_queue_rcv_skb(%struct.sock*, %struct.sk_buff*)

declare void %arp_solicit(%struct.neighbour*, %struct.sk_buff*)

declare void %arp_error_report(%struct.neighbour*, %struct.sk_buff*)

declare uint %arp_hash(sbyte*, %struct.net_device*)

declare int %arp_constructor(%struct.neighbour*)

declare void %parp_redo(%struct.sk_buff*)

declare int %inetdev_event(%struct.notifier_block*, uint, sbyte*)

declare int %inet_setsockopt(%struct.socket*, int, int, sbyte*, int)

declare int %inet_getsockopt(%struct.socket*, int, int, sbyte*, int*)

declare int %inet_listen(%struct.socket*, int)

declare int %inet_create(%struct.socket*, int)

declare int %inet_release(%struct.socket*)

declare int %inet_bind(%struct.socket*, %struct.sockaddr*, int)

declare int %inet_dgram_connect(%struct.socket*, %struct.sockaddr*, int, int)

declare int %inet_stream_connect(%struct.socket*, %struct.sockaddr*, int, int)

declare int %inet_accept(%struct.socket*, %struct.socket*, int)

declare int %inet_getname(%struct.socket*, %struct.sockaddr*, int*, int)

declare int %inet_recvmsg(%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)

declare int %inet_sendmsg(%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)

declare int %inet_shutdown(%struct.socket*, int)

declare int %inet_ioctl(%struct.socket*, uint, uint)

declare int %ipv4_sysctl_forward(%struct.ctl_table*, int, %struct.file*, sbyte*, uint*)

declare int %ipv4_sysctl_forward_strategy(%struct.ctl_table*, int*, int, sbyte*, uint*, sbyte*, uint, sbyte**)

declare int %proc_dointvec_minmax(%struct.ctl_table*, int, %struct.file*, sbyte*, uint*)

declare int %sysctl_intvec(%struct.ctl_table*, int*, int, sbyte*, uint*, sbyte*, uint, sbyte**)

void %get_current657() {
entry:
	unreachable
}

declare int %fib_inetaddr_event(%struct.notifier_block*, uint, sbyte*)

declare int %fib_netdev_event(%struct.notifier_block*, uint, sbyte*)

declare int %unix_listen(%struct.socket*, int)

declare int %unix_create(%struct.socket*, int)

declare int %unix_release(%struct.socket*)

declare int %unix_bind(%struct.socket*, %struct.sockaddr*, int)

declare int %unix_dgram_connect(%struct.socket*, %struct.sockaddr*, int, int)

declare int %unix_stream_connect(%struct.socket*, %struct.sockaddr*, int, int)

declare int %unix_socketpair(%struct.socket*, %struct.socket*)

declare int %unix_accept(%struct.socket*, %struct.socket*, int)

declare int %unix_getname(%struct.socket*, %struct.sockaddr*, int*, int)

declare int %unix_dgram_sendmsg(%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)

declare int %unix_stream_sendmsg(%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)

declare int %unix_dgram_recvmsg(%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)

declare int %unix_stream_recvmsg(%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)

declare int %unix_shutdown(%struct.socket*, int)

declare int %unix_ioctl(%struct.socket*, uint, uint)

declare uint %unix_poll(%struct.file*, %struct.socket*, %struct.poll_table_struct*)

declare int %packet_sendmsg_spkt(%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)

declare int %packet_sendmsg(%struct.socket*, %struct.msghdr*, int, %struct.scm_cookie*)

declare int %packet_release(%struct.socket*)

declare int %packet_bind_spkt(%struct.socket*, %struct.sockaddr*, int)

declare int %packet_bind(%struct.socket*, %struct.sockaddr*, int)

declare int %packet_recvmsg(%struct.socket*, %struct.msghdr*, int, int, %struct.scm_cookie*)

declare int %packet_getname_spkt(%struct.socket*, %struct.sockaddr*, int*, int)

declare int %packet_getname(%struct.socket*, %struct.sockaddr*, int*, int)

declare int %packet_setsockopt(%struct.socket*, int, int, sbyte*, int)

declare int %packet_getsockopt(%struct.socket*, int, int, sbyte*, int*)

declare int %packet_ioctl(%struct.socket*, uint, uint)
