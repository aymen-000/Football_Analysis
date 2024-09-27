import numpy as np
import cv2 as cv

class ViewTransforms:
    def __init__(self) -> None:
        self.court_width = 68
        self.court_length = 23.32

        # Points in the original image
        self.pixel_vert = np.array(
            [
                [110, 1035],
                [265, 275],
                [910, 260],
                [1640, 915]
            ], dtype=np.float32
        )

        # Corresponding points in the target transformed view
        self.target_vert = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ], dtype=np.float32)

        # Perspective transformation matrix
        self.perspective = cv.getPerspectiveTransform(self.pixel_vert, self.target_vert)

    def transform_point(self, point):
        # Convert the point to integer (if necessary)
        p = (int(point[0]), int(point[1]))

        # Check if the point is inside the polygon formed by pixel_vert
        is_inside = cv.pointPolygonTest(self.pixel_vert, p, False) >= 0
        if not is_inside:
            return None

        # Reshape the point to the required format (1, 1, 2)
        reshaped_point = np.array(point, dtype=np.float32).reshape(1, 1, 2)

        # Apply the perspective transformation
        transformed_point = cv.perspectiveTransform(reshaped_point, self.perspective)

        # Reshape the transformed point back to (2,) format
        return transformed_point.reshape(2)

    def add_position_transform_to_track(self, tracks):
        for obj, object_track in tracks.items():
            for frame_num, frame_info in enumerate(object_track):
                for track_id, track_info in frame_info.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)

                    # Transform the position
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        # Squeeze to remove extra dimensions and convert to list
                        position_transformed = position_transformed.squeeze().tolist()

                    # Add the transformed position to the track info
                    tracks[obj][frame_num][track_id]["position_transform"] = position_transformed
