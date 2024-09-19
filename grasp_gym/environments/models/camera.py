import cv2
import numpy as np
import pybullet as p

class GripperCamera():
    """
    Class to handle the camera of the robot gripper
    """

    def __init__(self, gripper, cube_id, render=False) -> None:
        self.gripper = gripper
        self.render = render
        self.cube_id = cube_id

        self.img_width = 224
        self.img_height = 224

    def visualize_camera(self, cam_pos):
        """
        Args:
            cam_pos (list): The position of the camera
        """

        pos = [cam_pos[0]+0.05, cam_pos[1], cam_pos[2]+0.05]

        # delete visual marker object
        if self.visual_marker_id != -1:
            p.removeBody(self.visual_marker_id)
            self.visual_marker_id = -1

        self.visual_marker_id = p.loadURDF("sphere_small.urdf", basePosition=pos, baseOrientation=[0, 0, 0, 1],
                              useFixedBase=True)

    def compute_matrices(self):
        """
        Compute the camera view and projection matrices

        Returns:
            projection_matrix (list): The projection matrix of the camera
            view_matrix (list): The view matrix of the camera
        """

        cam_pos = self.get_cam_pos() + np.array([0.05,0,0.0])
        cam_ori = self.get_cam_ori()
        cam_pitch = -90-cam_ori[1]

        # Get camera view and projection matrices
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_pos,
            distance=0.001,
            yaw=-90,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=80,
            aspect=1.0,
            nearVal=0.005,
            farVal=1.0
        )


        return projection_matrix, view_matrix
    
    def get_depth_image(self, gripper_status):
        """
        Get the depth image from the camera with bounding box around the target object

        Args:
            gripper_status (int): The status of the gripper (0: closed, 1: open)
        
        Returns:
            bb_depth_norm (np.array): The normalized depth image with bounding box
        """


        # Get the camera matrices
        projection_matrix, view_matrix = self.compute_matrices()

        # Get depth image
        width, height, rgb, depth_image, seg_mask = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        # Convert rgb to grey
        grey_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # Convert depth image to numpy array
        depth_array = np.array(depth_image, dtype=np.float32).reshape((height, width))

        # Normalize depth image
        min_depth = np.min(depth_array)
        max_depth = np.max(depth_array)
        range_depth = max_depth - min_depth

        if range_depth > 0:
            depth_normalized = (depth_array - min_depth) / range_depth
        else:
            # If range is zero, set the normalized depth to zero
            depth_normalized = np.zeros_like(depth_array, dtype=np.float32)

        # Draw Bounding Box around target object
        bb_depth_norm = self.draw_bounding_box(depth_normalized, seg_mask, gripper_status)

        # Resize image
        bb_depth_norm = cv2.resize(bb_depth_norm, (94, 94))

        if self.render: self.render_image(bb_depth_norm, grey_img)

        return bb_depth_norm

    def draw_bounding_box(self, img, seg_mask, gripper_status):
        """
        Draw bounding box around the target object in the depth image

        Args:
            img (np.array): The depth image
            seg_mask (np.array): The segmentation mask of the target object
            gripper_status (int): The status of the gripper (0: closed, 1: open)

        Returns:
            bb_img (np.array): The depth image with bounding box around the target object
        """
        seg_arr = np.equal(seg_mask, self.cube_id)
        seg_img = np.asarray(seg_arr, dtype=np.float32).reshape(self.img_height, self.img_width)

        # Get coordinates of the object in the segmentation image
        coords = np.argwhere(seg_img > 0)

        if coords.size == 0:
            return img

        min_corner = coords.min(axis=0)
        max_corner = coords.max(axis=0)

        bb_col = 0
        if gripper_status == 1:
            bb_col = 150

        bb_img = cv2.rectangle(img, (min_corner[1] - 3, min_corner[0] - 3), 
                                        (max_corner[1] + 3, max_corner[0] + 3), bb_col, 2)
        return bb_img
      
    def render_image(self, depth):
        """
        Args:
            depth (np.array): The depth image to render
        """
        cv2.imshow('Depth Image', depth)
        cv2.waitKey(3)

    def get_cam_pos(self):
        """
        Returns:
            cam_pos (np.array): The position of the camera in world coordinates
        """
        cam_pos, _ = p.getLinkState(self.gripper, 9)[:2]
        return np.array(cam_pos)
    
    def get_cam_ori(self):
        """
        Returns:
            cam_ori (np.array): The orientation of the camera in euler angles
        """

        _, cam_ori = p.getLinkState(self.gripper, 9)[:2]

        # Convert quaternion to euler angles
        cam_ori = (np.array(p.getEulerFromQuaternion(cam_ori))*180)/np.pi

        return np.array(cam_ori)