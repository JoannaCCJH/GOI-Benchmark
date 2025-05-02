import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt

def numpy_to_ply(coords, segments, output_path):
    
    # Get the unique segment labels (from first column)
    unique_labels = np.unique(segments[:, 0]).astype(int)
    print("Number of Unique labels: ", len(unique_labels))
    print(unique_labels)
    
    # Create a colormap - map each unique label to a distinct color
    # Using matplotlib's colormap to generate visually distinct colors
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    
    # Create a mapping from label to RGB color
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        rgba = cmap(i)  # Returns (r, g, b, a) in 0-1 range
        # Convert to 0-255 range and drop alpha channel
        label_to_color[label] = np.array([rgba[0], rgba[1], rgba[2]]) * 255
        
    # Combine coordinates and segment labels into a structured array
    # Assuming coords are (x,y,z) and segments first column is the label
    vertex_data = np.zeros(coords.shape[0], 
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    vertex_data['x'] = coords[:, 0]
    vertex_data['y'] = coords[:, 1]
    vertex_data['z'] = coords[:, 2]
    
    # Set colors based on segment labels
    for i in range(coords.shape[0]):
        label = int(segments[i, 0])
        color = label_to_color[label]
        vertex_data['red'][i] = color[0]
        vertex_data['green'][i] = color[1]
        vertex_data['blue'][i] = color[2]
    
    # Create PlyElement
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Create and write PlyData to file
    ply_data = PlyData([vertex_element], text=True)  # Use text=False for binary format
    ply_data.write(output_path)
    
    print(f"PLY file saved to {output_path}")

if __name__ == "__main__":
    
    coord_path = "/scratch/joanna_cheng/scannetpp_gt/val/0d2ee665be/coord.npy"
    segment_path = "/scratch/joanna_cheng/scannetpp_gt/val/0d2ee665be/segment.npy"
    output_path = "./0d2ee665be_gt_point_cloud.ply"
    
    coord = np.load(coord_path)               # (N, 3)
    segment = np.load(segment_path)           # (N, 3) first col => label index
    
    numpy_to_ply(coord, segment, output_path)
    