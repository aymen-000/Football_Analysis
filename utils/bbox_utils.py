def get_center(bbox) : 
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2) , int((y1+y2)/2)

def get_bbox_width(bbox) : 
    return bbox[2]-bbox[0]

def get_distance(p1, p2):
    return ((int(p1[0]) - int(p2[0]))**2 + (int(p1[1]) - int(p2[1]))**2)**0.5

def get_foot_position(bbox) : 
    x1,y1,x2,y2 = bbox
    return (int((x1+x2)/2) , y2)
