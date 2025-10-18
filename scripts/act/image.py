'''
if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 获取原始图像尺寸
        width, height = image.size
        if debug:
            print(f"原始图像尺寸: {width}x{height}")

        if index not in (0, 1):
            raise ValueError(f"unsupported index: {index}")
        
        # 两个视角都使用相同的裁剪方式：从特定位置裁剪到360*360
        left = 180
        right = 540
        top = 0
        bottom = 360
        if debug:
            print(f"cam_{index} 640x480图片，裁剪区域: ({left}, {top}, {right}, {bottom})")
        
        # 裁剪
        image_cropped = image.crop((left, top, right, bottom))
        if debug:
            print(f"裁剪后尺寸: {image_cropped.size}")

        # 缩放到目标尺寸（确保能被16整除，DINOv3 patch_size=16）
        target_size = (224, 224)  # 224能被16整除
        image_resized = image_cropped.resize(target_size, Image.Resampling.LANCZOS)
        if debug:
            print(f"缩放后尺寸: {image_resized.size}")


if isinstance(image, np.ndarray):
            # 保存numpy格式用于cam_1的padding操作
            image_np = image
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            image_np = np.array(image)
        
        # 获取原始图像尺寸
        width, height = image_pil.size
        if debug:
            print(f"原始图像尺寸: {width}x{height}")

        if index not in (0, 1):
            raise ValueError(f"unsupported index: {index}")
        
        # 目标尺寸（确保能被16整除，DINOv3 patch_size=16）
        target_size = (224, 224)
        
        if index == 0:
            # cam_0（固定机位）：裁剪处理
            left = 180
            right = 540
            top = 0
            bottom = 360
            if debug:
                print(f"cam_0 裁剪区域: ({left}, {top}, {right}, {bottom})")
            
            # 裁剪
            image_processed = image_pil.crop((left, top, right, bottom))
            if debug:
                print(f"裁剪后尺寸: {image_processed.size}")
        
        else:  # index == 1
            # cam_1（下视相机）：扩张像素处理
            if debug:
                print(f"cam_1 扩张像素处理")
            
            # 使用cv2进行padding（上下各扩展80像素）
            image_expanded = cv2.copyMakeBorder(
                image_np,
                80, 80, 0, 0,  # top, bottom, left, right
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # 黑色 (RGB格式)
            )
            if debug:
                print(f"扩展后尺寸: {image_expanded.shape}")
            
            # 转换回PIL格式
            image_processed = Image.fromarray(image_expanded)

        # 缩放到目标尺寸
        image_resized = image_processed.resize(target_size, Image.Resampling.LANCZOS)
        if debug:
            print(f"缩放后尺寸: {image_resized.size}")           
'''