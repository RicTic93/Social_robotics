config = {
    'num_objects': 3,    # Number of objects in the scene

    'dynamic_humans_count': 3,     # Number of dynamic humans
    'static_groups':[              # Parameters for static groups
        {
            'count': 3,
            'formation': 'converging',
            'center_pos': [3.0, 7.0],
            'radius': 1.0,
        },
        {
            'count': 2,
            'formation': 'diverging',
            'center_pos': [7.0, 3.0],
            'radius': 1.0,
        }
            ],
                
    'max_steps': 500,    # Maximum number of steps per episode

    'social_distance': 1.5   # Distance entre les humans
}
