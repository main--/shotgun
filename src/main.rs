// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// For the purpose of this example all unused code is allowed.
#![allow(dead_code)]


//#![windows_subsystem = "windows"]


extern crate cgmath;
extern crate image;
extern crate winit;
extern crate screenshot;

#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate vulkano_win;

#[macro_use]
extern crate cfg_if;

use vulkano_win::VkSurfaceBuild;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

fn main() {
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let extensions = vulkano_win::required_extensions();
    let instance = vulkano::instance::Instance::new(None, &extensions, &[]).expect("failed to create instance");

    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new().with_fullscreen(Some(events_loop.get_primary_monitor())).build_vk_surface(&events_loop, instance.clone()).unwrap();

    let mut dimensions;

    let queue = physical.queue_families().find(|&q| q.supports_graphics() &&
                                                   window.surface().is_supported(q).unwrap_or(false))
                                                .expect("couldn't find a graphical queue family");

    let device_ext = vulkano::device::DeviceExtensions {
        khr_swapchain: true,
        .. vulkano::device::DeviceExtensions::none()
    };
    let (device, mut queues) = vulkano::device::Device::new(physical, physical.supported_features(),
                                                            &device_ext, [(queue, 0.5)].iter().cloned())
                               .expect("failed to create device");
    let queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = window.surface().capabilities(physical).expect("failed to get surface capabilities");

        dimensions = caps.current_extent.unwrap_or([1024, 768]);
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;

        vulkano::swapchain::Swapchain::new(device.clone(), window.surface().clone(), caps.min_image_count,
                                           format, dimensions, 1,
                                           usage, &queue, vulkano::swapchain::SurfaceTransform::Identity,
                                           alpha,
                                           vulkano::swapchain::PresentMode::Fifo, true, None).expect("failed to create swapchain")
    };


    #[derive(Debug, Clone)]
    struct Vertex2 { position: [f32; 2] }
    impl_vertex!(Vertex2, position);

    let mut select_orig = (0., 0.); //(-0.5, -0.5);
    let mut select_targ = (0., 0.); //(0.5, 0.5);
    let mut select_cache = (0., 0.);
    let mut selecting = false;

    let vertex_buffer2 = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[Vertex2]>
        ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(),
                    [
                        Vertex2 { position: [select_orig.0, select_orig.1] },
                        Vertex2 { position: [select_targ.0, select_orig.1] },
                        Vertex2 { position: [select_targ.0, select_targ.1] },
                        Vertex2 { position: [select_orig.0, select_targ.1] },
                        Vertex2 { position: [select_orig.0, select_orig.1] },
                    ].iter().cloned()).expect("failed to create buffer 2");

    let vs2 = vs2::Shader::load(device.clone()).expect("failed to create shader module");
    let fs2 = fs2::Shader::load(device.clone()).expect("failed to create shader module");

    #[derive(Debug, Clone)]
    struct Vertex { position: [f32; 2], overlay: [f32; 4] }
    impl_vertex!(Vertex, position, overlay);

    let vertex_buffer = vulkano::buffer::cpu_access::CpuAccessibleBuffer::<[Vertex]>
                               ::from_iter(device.clone(), vulkano::buffer::BufferUsage::all(),
                                       [
                                           Vertex { position: [-1., -1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                                           Vertex { position: [-1.,  1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                                           Vertex { position: [ 1., -1. ], overlay: [0.8, 0.8, 0.8, 0.4] },

                                           Vertex { position: [-1.,  1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                                           Vertex { position: [ 1., -1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                                           Vertex { position: [ 1.,  1. ], overlay: [0.8, 0.8, 0.8, 0.4] },

                                           /* initially, no selection. so zero out the selection vertices
                                           Vertex { position: [select_orig.0, select_orig.1], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [select_orig.0, select_targ.1], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [select_targ.0, select_orig.1], overlay: [0., 0., 0., 0.] },

                                           Vertex { position: [select_orig.0, select_targ.1], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [select_targ.0, select_orig.1], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [select_targ.0, select_targ.1], overlay: [0., 0., 0., 0.] },
                                            */
                                           Vertex { position: [0., 0.], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [0., 0.], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [0., 0.], overlay: [0., 0., 0., 0.] },

                                           Vertex { position: [0., 0.], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [0., 0.], overlay: [0., 0., 0., 0.] },
                                           Vertex { position: [0., 0.], overlay: [0., 0., 0., 0.] },
                                       ].iter().cloned()).expect("failed to create buffer");

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    let renderpass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: DontCare,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()
    );

    let (texture, tex_future) = {
        //let image = image::load_from_memory_with_format(include_bytes!("grab.png"),
        //                                                image::ImageFormat::PNG).unwrap().to_rgba();
        //let image_data = image.into_raw().clone();

        //        let image_data = include_bytes!("grab.bin");
        #[cfg(not(feature = "sway"))]
        let screenshot = screenshot::get_screenshot(0).unwrap();
        #[cfg(not(feature = "sway"))]
        let image_data = screenshot.as_ref();
        #[cfg(feature = "sway")]
        let image_data = std::process::Command::new("/usr/bin/swaygrab").arg("--raw").output().unwrap().stdout;

        #[cfg(not(feature = "sway"))]
        let format = vulkano::format::B8G8R8A8Unorm;
        #[cfg(feature = "sway")]
        let format = vulkano::format::R8G8B8A8Srgb;

        vulkano::image::immutable::ImmutableImage::from_iter(
            image_data.iter().cloned(),
            vulkano::image::Dimensions::Dim2d { width: 1920, height: 1080 },
            format,
            queue.clone()).unwrap()
    };


    let sampler = vulkano::sampler::Sampler::new(device.clone(), vulkano::sampler::Filter::Linear,
                                                 vulkano::sampler::Filter::Linear, vulkano::sampler::MipmapMode::Nearest,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 vulkano::sampler::SamplerAddressMode::Repeat,
                                                 0.0, 1.0, 0.0, 0.0).unwrap();

#[cfg(target_os = "windows")]
let do_flip = 0;
#[cfg(not(target_os = "windows"))]
let do_flip = 1; // TODO: only if sway

    let pipeline = Arc::new(vulkano::pipeline::GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
                            .vertex_shader(vs.main_entry_point(), vs::SpecializationConstants { flip: do_flip })
                            .triangle_list()
                            //.triangle_strip()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .blend_alpha_blending()
        .render_pass(vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap())
        .build(device.clone())
                            .unwrap());

    let pipeline2 = Arc::new(vulkano::pipeline::GraphicsPipeline::start()
                             .vertex_input_single_buffer::<Vertex2>()
                             .vertex_shader(vs2.main_entry_point(), ())
                             .line_strip()
                             .viewports_dynamic_scissors_irrelevant(1)
                             .fragment_shader(fs2.main_entry_point(), ())
                             .blend_pass_through()
                             .render_pass(vulkano::framebuffer::Subpass::from(renderpass.clone(), 0).unwrap()) // FIXME
                             .build(device.clone()).unwrap());

    let set = Arc::new(vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(pipeline.clone(), 0)
        .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
        .build().unwrap()
    );

    let mut framebuffers: Option<Vec<Arc<vulkano::framebuffer::Framebuffer<_,_>>>> = None;

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Box::new(tex_future) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();
        if recreate_swapchain {

            dimensions = window.surface().capabilities(physical)
                .expect("failed to get surface capabilities")
                .current_extent.unwrap_or([1024, 768]);

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(vulkano::swapchain::SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                },
                Err(err) => panic!("{:?}", err)
            };

            std::mem::replace(&mut swapchain, new_swapchain);
            std::mem::replace(&mut images, new_images);

            framebuffers = None;

            recreate_swapchain = false;
        }

        if framebuffers.is_none() {
            let new_framebuffers = Some(images.iter().map(|image| {
                Arc::new(vulkano::framebuffer::Framebuffer::start(renderpass.clone())
                         .add(image.clone()).unwrap()
                         .build().unwrap())
            }).collect::<Vec<_>>());
            std::mem::replace(&mut framebuffers, new_framebuffers);
        }

        let (image_num, future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(vulkano::swapchain::AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let cb = vulkano::command_buffer::AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
            .unwrap()
            .begin_render_pass(
                framebuffers.as_ref().unwrap()[image_num].clone(), false,
                vec![]).unwrap()
            .draw(pipeline.clone(),
                   vulkano::command_buffer::DynamicState {
                      line_width: None,
                      viewports: Some(vec![vulkano::pipeline::viewport::Viewport {
                          origin: [0.0, 0.0],
                          dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                          depth_range: 0.0 .. 1.0,
                      }]),
                      scissors: None,
                  },
                  vertex_buffer.clone(),
                  set.clone(), ()).unwrap()
            .draw(pipeline2.clone(),
                   vulkano::command_buffer::DynamicState {
                      line_width: None,
                      viewports: Some(vec![vulkano::pipeline::viewport::Viewport {
                          origin: [0.0, 0.0],
                          dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                          depth_range: 0.0 .. 1.0,
                      }]),
                      scissors: None,
                  },
                  vertex_buffer2.clone(),
                  set.clone(), ()).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        let future = previous_frame_end.join(future)
            .then_execute(queue.clone(), cb).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush().unwrap();
        previous_frame_end = Box::new(future) as Box<_>;

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => done = true,
                winit::Event::WindowEvent { event: winit::WindowEvent::Resized(_, _), .. } => recreate_swapchain = true, // nvidia quirk: driver doesn't notice this
                winit::Event::WindowEvent { event: winit::WindowEvent::KeyboardInput { input: winit::KeyboardInput { virtual_keycode: Some(winit::VirtualKeyCode::Escape), .. }, .. }, .. } => {
// quit
done = true;
                }
                winit::Event::WindowEvent { event: winit::WindowEvent::MouseInput { state, button: winit::MouseButton::Left, .. }, .. } => {
                    let state = state == winit::ElementState::Pressed;
                    if state != selecting {
                        selecting = state;
                        if selecting {
                            select_orig = select_cache;
                            select_targ = select_cache;
                        }
                    }
                }
                winit::Event::WindowEvent { event: winit::WindowEvent::CursorMoved { position: (w, h), .. }, .. } => {
                    let c1 = ((w as f32) / (dimensions[0] as f32)) * 2. - 1. + 0.001;
                    let c2 = ((h as f32) / (dimensions[1] as f32)) * 2. - 1. + 0.001;
                    if selecting {
                        select_targ = (c1, c2);
                    } else {
                        select_cache = (c1, c2);
                        return;
                    }

                    if previous_frame_end.queue().is_some() {
                    let wf = std::mem::replace(&mut previous_frame_end, Box::new(vulkano::sync::now(device.clone())));
                        wf.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
                    }

                    {
                        let mut wl1 = vertex_buffer.write().unwrap();
                        for (t, d) in wl1.iter_mut().zip([
                            Vertex { position: [-1., -1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                            Vertex { position: [-1.,  1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                            Vertex { position: [ 1., -1. ], overlay: [0.8, 0.8, 0.8, 0.4] },

                            Vertex { position: [-1.,  1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                            Vertex { position: [ 1., -1. ], overlay: [0.8, 0.8, 0.8, 0.4] },
                            Vertex { position: [ 1.,  1. ], overlay: [0.8, 0.8, 0.8, 0.4] },


                            Vertex { position: [select_orig.0, select_orig.1], overlay: [0., 0., 0., 0.] },
                            Vertex { position: [select_orig.0, select_targ.1], overlay: [0., 0., 0., 0.] },
                            Vertex { position: [select_targ.0, select_orig.1], overlay: [0., 0., 0., 0.] },

                            Vertex { position: [select_orig.0, select_targ.1], overlay: [0., 0., 0., 0.] },
                            Vertex { position: [select_targ.0, select_orig.1], overlay: [0., 0., 0., 0.] },
                            Vertex { position: [select_targ.0, select_targ.1], overlay: [0., 0., 0., 0.] },
                            ].iter()) {
                            *t = d.clone();
                        }
                    }
                    {
                        let mut wl2 = vertex_buffer2.write().unwrap();
                        for (t, d) in wl2.iter_mut().zip([
                            Vertex2 { position: [select_orig.0, select_orig.1] },
                            Vertex2 { position: [select_targ.0, select_orig.1] },
                            Vertex2 { position: [select_targ.0, select_targ.1] },
                            Vertex2 { position: [select_orig.0, select_targ.1] },
                            Vertex2 { position: [select_orig.0, select_orig.1] },
                            /*
                            Vertex2 { position: [-0.5, -0.5] },
                            Vertex2 { position: [ 0.5, -0.5] },
                            Vertex2 { position: [ 0.5,  0.5] },
                            Vertex2 { position: [-0.5,  0.5] },
                            Vertex2 { position: [-0.5, -0.5] },
*/
                            ].iter()) {
                            *t = d.clone();
                        }
                    }
                }
                _ => ()
            }
        });
        if done { return; }
    }
}

mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 position;
layout(location = 2) in vec4 overlay;
layout(location = 0) out vec2 tex_coords;
layout(location = 2) out vec4 tex_overlay;

layout(constant_id = 0) const bool flip = false;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);

    tex_coords = position * 0.5; // * vec2(1, -1);
    if (flip) tex_coords *= vec2(1, -1);
    tex_coords += vec2(0.5);

    tex_overlay = overlay;
}
"]
    struct Dummy;
}

mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 2) in vec4 tex_overlay;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    f_color = mix(texture(tex, tex_coords), tex_overlay, tex_overlay.a);
    f_color.a = 1.0;
}
"]
    struct Dummy;
}


mod vs2 {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"]
    struct Dummy;
}

mod fs2 {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1, 0, 0, 1);
}
"]
    struct Dummy;
}
