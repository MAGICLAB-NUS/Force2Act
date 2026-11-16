# USB-A manufacturer CAD

Downloaded 2026-10-06 from Würth Elektronik's [WR-USB product catalogue](https://www.we-online.com/en/components/products/INPUT_OUTPUT_WR_USB_STANDARD).

| Part | Role | Original CAD | Drawing |
|---|---|---|---|
| 61400416021 | USB 2.0 Type-A receptacle | [STEP](https://www.we-online.com/components/products/download/Download_STP_61400416021%20%28rev1%29.stp) | [Datasheet](https://www.we-online.com/components/products/datasheet/61400416021.pdf) |
| 629004113921 | USB 2.0 Type-A plug | [STEP](https://www.we-online.com/components/products/download/Download_STP_629004113921%20%28rev1%29.stp) | [Datasheet](https://www.we-online.com/components/products/datasheet/629004113921.pdf) |

The linked source CAD and datasheets remain under the manufacturer's ownership and terms; no open-source CAD licence is asserted here. These are manufacturer component models rather than a scanned cable/PC enclosure. A simulated rigid holder attaches the plug to the Rizon4S.

STEP SHA-256:

```text
61400416021.stp  688cda3a93dfff78592001d70977a89431b8b6dd91ddcc6c8ff3c318fe0ac53d
629004113921.stp ce490dc913de8cc27fa12b07123162d7f71357b377708fee4a005accb8dadfb5
```

The STEP reader reports one unresolved reference in the receptacle export. All six solids are transferred; the receptacle shield tessellation is not watertight. The complete meshes are used for visuals only. Collision never uses a convex hull of a hollow shell or the non-watertight shield.

Source CAD units are millimetres; simulation mesh coordinates are in metres. Twelve `socket_XX.obj` / `plug_XX.obj` meshes retain all transferred solids: plastic, four contacts and shielding.

Separate convex prisms use the socket shell/tongue sections at CAD y = −3 mm and the plug shell/insert at y = 0. Independent rear stops approximate plastic blocks. The shell cavity centre is CAD z = 0.48 mm. The plug tip is CAD y = 7.375 mm, placed 245 mm from the flange. The receptacle front plane is CAD y = −11.3260125 mm; its rear root is y = 2.675 mm, giving a 14.0010125 mm local entrance height. The scene mounts that root 40 mm above the table and commands an 8.5 mm mating depth.

Correct mating flips the plug's plastic insert opposite the receptacle tongue. The two rigid mating cross sections have zero overlap in that orientation; a 180° reversal has about 12 mm² overlap. This geometric obstruction generates solver contact forces. No orientation-dependent force rule is added.

Rigid proxies omit elastic spring fingers, shell retention holes, front flares and detailed chamfers. The tongue front is truncated to CAD y = −10.8 mm and the straight shell starts at −10.975 mm. Friction 0.3, holder geometry/mass, 40 mm mount and 8.5 mm goal are simulation design choices. Free aligned insertion can have zero force because the proxies have clearance; this does not represent a measured real USB insertion-force profile or verify electrical connectivity.

The simulator loads the supplied URDFs and meshes directly. No CAD conversion
or asset generation is required to run the scene.
