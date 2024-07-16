import cv2
import numpy as np
import pybullet as p

class GripperCamera():

    def __init__(self, gripper, cube_id, render=False) -> None:
        self.gripper = gripper
        self.render = render
        self.cube_id = cube_id

        self.img_width = 224
        self.img_height = 224

    def visualize_camera(self, cam_pos, cam_ori):

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
        """

       
        cam_pos = self.get_cam_pos() + np.array([0.028,0,0.0])
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
    
    def get_depth_image(self):

        # Get the camera matrices
        projection_matrix, view_matrix = self.compute_matrices()

        # Get depth image
        width, height, rgb, depth_image, seg_mask = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        # Convert depth image to numpy array
        depth_array = np.array(depth_image).reshape((height, width))

        # Normalize depth image
        depth_normalized = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))

        # Draw Bounding Box around target object
        bb_depth_norm = self.draw_bounding_box(depth_normalized, seg_mask, 1)

        if self.render: self.render_image(bb_depth_norm, rgb)

        return depth_normalized

    def draw_bounding_box(self, img, seg_mask, obj_id):

        seg_arr = np.equal(seg_mask, obj_id)

        # reshape segmented array to image
        seg_img = np.asarray(seg_arr, dtype=np.float32).reshape(self.img_height, self.img_width)

        # Get Maximum and Minimum conrer points
        min_corner = [990,990]
        max_corner = [0,0]

        for x in range(self.img_height):
            for y in range(self.img_width):
                cell = seg_img[x][y]

                if cell > 0 and x < min_corner[1]:
                    min_corner[1] = x
                if cell > 0 and x > max_corner[1]:
                    max_corner[1] = x
                if cell > 0 and y < min_corner[0]:
                    min_corner[0] = y
                if cell > 0 and y > max_corner[0]:
                    max_corner[0] = y

        if min_corner[0] != 990 and min_corner[1] != 990 and max_corner[0] != 0 and max_corner[1] != 0 and \
            min_corner[0]-3 >= 0 and min_corner[1]-3 >= 0 and max_corner[0]+3 <= 224 and max_corner[1]+3 <= 224:
            
            bb_img = cv2.rectangle(img, (min_corner[0]-3,min_corner[1]-3), (max_corner[0]+3,max_corner[1]+3), 0, 2)
            return bb_img
        
        else:
            return img
        
    def render_image(self, depth, rgb):
        """
        Render the depth and rgb images
        """
        cv2.imshow('Depth Image', depth)
        cv2.imshow('RGB Image', rgb)
        cv2.waitKey(3)

    def get_cam_pos(self):
        """
        Get the position of the camera
        """
        cam_pos, _ = p.getLinkState(self.gripper, 9)[:2]
        return np.array(cam_pos)
    
    def get_cam_ori(self):
        """
        Get the orientation of the camera in euler angles (deg)
        """

        _, cam_ori = p.getLinkState(self.gripper, 9)[:2]

        # Convert quaternion to euler angles
        cam_ori = (np.array(p.getEulerFromQuaternion(cam_ori))*180)/np.pi

        return np.array(cam_ori)