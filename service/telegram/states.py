from aiogram.fsm.state import StatesGroup, State


class ProjectStates(StatesGroup):
    train_type_select = State()
    model_select = State()
    hyperparam_select = State()
    hyperparam_value = State()
    time_limit_input = State()
    model_id_input = State()
    load_file = State()
    training = State()
    model_remove = State()
    model_load = State()
