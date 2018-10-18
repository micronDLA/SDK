# Questions and answers

Q: Issue: Can't find FPGA card

A: Make sure the picocomputing-6.0.0.21 release is installed properly. Please run the following commands. It should print the following outputs.
```
lspci | grep -i pico
    05:00.0 Memory controller: Pico Computing Device 0045 (rev 05)
    08:00.0 Memory controller: Pico Computing Device 0510 (rev 05)
lsmod | grep -i pico
    pico                 3493888  12
dmesg | grep -i pico
[   12.030836] pico: loading out-of-tree module taints kernel.
[   12.031521] pico: module verification failed: signature and/or required key missing - tainting kernel
[   12.035737] pico:init_pico(): Pico driver 5.0.9.18 compiled on Mar  1 2018 at 17:22:20
[   12.035739] pico:init_pico(): debug level: 3
[   12.035751] pico:init_pico(): got major number 240
[   12.035797] pico:pico_init_e17(): id: 19de:45 19de:2045 5
[   12.035798] pico:pico_init_v6_v5(): id: 19de:45 19de:2045 5
[   12.035806] pico 0000:05:00.0: enabling device (0100 -> 0102)
[   12.035883] pico:pico_init_v6_v5(): fpga 0 assigned to dev_table[1] (addr: 0xffffffffc0a2f2a8). minor=224
[   12.035919] pico:pico_init_v6_v5(): bar 0 at 0xffffa2b9c5f00000 for 0x100000 bytes
[   12.035938] pico:pico_init_8664(): Initializing backplane: 0xffff945549cb2300
[   12.036205] pico:init_jtag(): Initializing JTAG: Backplane (0x8780) (backplane ID: 0x700)
[   12.036206] pico:init_jtag(): Using ex700 Spartan image
[   12.036445] pico:init_jtag(): Initializing JTAG: Module (0x45) (backplane ID: 0x700)
[   12.036446] pico:init_jtag(): Using ex700 Spartan image
[   12.036446] pico:pico_init_v6_v5(): writing 1 to 0x10 to enable stream machine
[   12.036452] pico:pico_init_v6_v5(): Firmware version (0x810): 0x5000708
[   12.036462] pico:update_fpga_cfg(): fpga version: 0x5000000 device: 0x45
[   12.037641] pico:update_fpga_cfg(): card 224 firmware version (from PicoBus): 0x5000708
[   12.039948] pico:update_fpga_cfg(): 0xFFE00050: 0x2020
[   12.039949] pico:update_fpga_cfg(): found a user picobus 32b wide
[   12.039950] pico:update_fpga_cfg(): cap: 0x410, widths: 32, 32
[   12.040121] pico:require_ex500_jtag(): S6 IDCODE: 0x44028093
[   12.040212] pico:require_ex500_jtag(): S6 USERCODE: 0x7000038
[   12.040685] pico:require_ex500_jtag(): S6 status: 0x3cec
[   12.040893] pico:pico_init_e17(): id: 19de:510 19de:2060 5
[   12.040894] pico:pico_init_v6_v5(): id: 19de:510 19de:2060 5
[   12.040899] pico 0000:08:00.0: enabling device (0100 -> 0102)
[   12.041115] pico:pico_init_v6_v5(): fpga 0 assigned to dev_table[2] (addr: 0xffffffffc0a2f2b0). minor=1
[   12.041131] pico:pico_init_v6_v5(): bar 0 at 0xffffa2b9c6100000 for 0x100000 bytes
[   12.041382] pico:init_jtag(): Initializing JTAG: Module (0x510) (backplane ID: 0x700)
[   12.041384] pico:pico_init_v6_v5(): creating device files for Pico FPGA #1 (fpga=0xffff9455483a8158 on card 0xffff9455483a8000)
[   12.041385] pico: creating device with class=0xffff94554054f480, major=240, minor=1
[   12.041421] pico:pico_init_v6_v5(): writing 1 to 0x10 to enable stream machine
[   12.041425] pico:pico_init_v6_v5(): Firmware version (0x810): 0x6000000
[   12.041430] pico:update_fpga_cfg(): fpga version: 0x5000000 device: 0x510
[   12.047453] pico:update_fpga_cfg(): detected non-virgin card (0x4000. probably from driver reload). disabling picobuses till the FPGA is reloaded.
[   12.047495] pico:pico_init_e17(): id: 19de:510 19de:2060 5
[   12.047497] pico:pico_init_v6_v5(): id: 19de:510 19de:2060 5
[   12.047502] pico 0000:09:00.0: enabling device (0100 -> 0102)
[   12.047699] pico:pico_init_v6_v5(): fpga 0 assigned to dev_table[3] (addr: 0xffffffffc0a2f2b8). minor=2
[   12.047722] pico:pico_init_v6_v5(): bar 0 at 0xffffa2b9c7000000 for 0x100000 bytes
[   12.047968] pico:init_jtag(): Initializing JTAG: Module (0x510) (backplane ID: 0x700)
```

Q: Can I run my own model?

A: yes, all models that are derivatives of the onles listed in the Supported Networks section can be modified and will run, within the limitations of the system.

Q: How can I create my own demonstration applications?

A: Just modify our example in the Demo section and you will be running in no time!

Q: How will developers be able to develop on your platform?

A: They will need to provide a neural network model only. No need to write any special code. FWDNXT will update the software periodically based on users and market needs.

Q:Will using FWDNXT inference engine require FPGA expertise? How much do I really have to know?

A: Nothing at all, it will be all transparent to users, just like using a GPU.

Q: How can I migrate my CUDA-based designs into FWDNXT inference engine?

A: FWDNXT inference engine offer its own optimized compiler, and you only need to specify trained model file

Q: What tools will I need at minimum?

A: FWDNXT inference engine on an FPGA and FWDNXT SDK tools

Q: What if my designs are in OpenCL or one of the FPGA vendor's tools?

A: FWDNXT inference engine will soon be available in OpenCL drivers

Q: Why should people want to develop on the FWDNXT inference engine platform?

A: Best performance per power and scalability, plus our hardware has a small form factor that can scale from single small module to high-performance systems

Q: How important is scalability? How does that manifest in terms of performance?

A: it is important when the application needs scale, or are not defined. Scalability allows the same application to run faster or in more devices with little or no work.
