# NEVER_COLLISIONS = [
#     ('panda_link0', 'panda_link1'),
#     ('panda_link1', 'panda_link2'),
#     ('panda_link2', 'panda_link3'),
#     ('panda_link3', 'panda_link4'),
#     ('panda_link4', 'panda_link5'),
#     ('panda_link5', 'panda_link6'),
#     ('panda_link6', 'panda_link7'),
#     ('panda_link7', 'panda_link8'),
#     ('panda_link8', 'panda_hand'),
#     ('panda_hand', 'panda_leftfinger'),
#     ('panda_hand', 'panda_rightfinger'),
#     ('panda_leftfinger', 'panda_rightfinger')
# ]

def generate_all_collision_pairs(links):
    return [(link1, link2) for i, link1 in enumerate(links) for link2 in links[i+1:]]

links = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3',
         'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7',
         'panda_link8', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']

NEVER_COLLISIONS = generate_all_collision_pairs(links)