"""Сommon handlers and registration"""
from aiogram import types
from aiogram.dispatcher import Dispatcher

from plugins.reaction import reaction


class CommonHandlers:
    """Сommon handlers"""

    async def start_command(message: types.Message) -> None:
        """
        Handler of the /start command

        Args:
            message (types.Message): Instance of the Message class.
        """

        await message.answer(
            'Let\'s get started!🔥'
        )

    async def help_command(message: types.Message) -> None: 
        """
        Handler of the /help command

        Args:
            message (types.Message): Instance of the Message class.
        """

        await message.answer(
            'We\'ll be there soon🆘'
            )

    async def neural_network_reaction(message: types.Message) -> None:
        """
        Handler for unknown commands or messages

        Args:
            message (types.Message): Instance of the Message class.
        """

        await message.answer(
            reaction(message.text)
            )

def register_client_handlers(dp: Dispatcher) -> None:
    """
    Registration of common handlers

    Args:
        dp (Dispatcher): Instance of the Dispatcher class.
    """
    dp.register_message_handler(CommonHandlers.start_command, commands=['start'])
    dp.register_message_handler(CommonHandlers.help_command, commands=['help'])

    dp.register_message_handler(CommonHandlers.neural_network_reaction)
