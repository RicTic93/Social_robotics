config = {
    'num_objects': 4,    # Number of objects in the scene

    'dynamic_humans_count': 1,     # Number of dynamic humans
    'static_groups':[              # Parameters for static groups
        {
            'count': 3,
            'formation': 'converging',
            'center_pos': [3.0, 5.0],
            'radius': 0.75,
        },
        {
            'count': 2,
            'formation': 'diverging',
            'center_pos': [7.0, 3.0],
            'radius':  0.75,
        },
        {
            'count': 2,
            'formation': 'converging',
            'center_pos': [6.5, 7.0],
            'radius': 0.75,
        }
        ],
                
    'max_steps': 500,    # Maximum number of steps per episode

    'social_distance': 1.5   # Distance between humans
}
